import modal
import zipfile
import os
import json
import time

app = modal.App("unzip-dataset-job")
vol = modal.Volume.from_name("gstdr-dataset-v3-50k")

image = modal.Image.debian_slim(python_version="3.11")

CHUNK_SIZE = 8000       # files per batch_upload call -- tune down if you see stream timeouts again
MANIFEST_PATH = "/tmp/extracted/.upload_manifest.json"


def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return set(json.load(f))
    return set()


def save_manifest(done_chunks):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(sorted(done_chunks), f)


def upload_chunk(idx, chunk, local_extract_dir):
    start = time.time()
    with vol.batch_upload(force=True) as batch:
        for rel_path in chunk:
            local_path = os.path.join(local_extract_dir, rel_path)
            batch.put_file(local_path, "/" + rel_path)
    elapsed = time.time() - start
    rate = len(chunk) / elapsed if elapsed > 0 else 0
    print(f"chunk {idx} done: {len(chunk)} files in {elapsed:.1f}s ({rate:.0f} files/s)", flush=True)


@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=3600,
    memory=8192,
    cpu=4,
)
def run_unzip():
    zip_path = "/data/dataset_v3_50k.zip"
    local_extract_dir = "/tmp/extracted"

    if not os.path.exists(zip_path):
        print(f"Error: Could not find {zip_path}")
        return

    print("Extracting zip to local container disk...")
    t0 = time.time()
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        names = zip_ref.namelist()
        print(f"zip contains {len(names)} entries")
        zip_ref.extractall(local_extract_dir)
    print(f"extraction took {time.time() - t0:.1f}s")

    extracted_rel = set()
    for root, _, files in os.walk(local_extract_dir):
        for fname in files:
            rel = os.path.relpath(os.path.join(root, fname), local_extract_dir)
            extracted_rel.add(rel.replace(os.sep, "/"))

    zip_file_entries = {n for n in names if not n.endswith("/")}
    print(f"extracted {len(extracted_rel)} files to local disk")

    missing = zip_file_entries - extracted_rel
    if missing:
        print(f"WARNING: {len(missing)} zip file-entries did not land on disk:")
        for m in sorted(missing)[:20]:
            print(f"  missing: {m}")
    else:
        dir_entries = len(names) - len(zip_file_entries)
        print(f"count 'mismatch' fully explained by {dir_entries} directory entries in zip -- no data loss")

    all_files = sorted(extracted_rel)
    chunks = {i: all_files[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
              for i in range((len(all_files) + CHUNK_SIZE - 1) // CHUNK_SIZE)}
    done_chunks = load_manifest()
    pending = {i: c for i, c in chunks.items() if i not in done_chunks}

    print(f"Uploading {len(all_files)} files in {len(chunks)} chunks, "
          f"{len(pending)} pending (sequential)", flush=True)
    if done_chunks:
        print(f"Resuming: {len(done_chunks)} chunk(s) already uploaded, skipping those", flush=True)

    t_upload = time.time()
    for idx in sorted(pending.keys()):
        upload_chunk(idx, pending[idx], local_extract_dir)
        done_chunks.add(idx)
        save_manifest(done_chunks)

    vol.commit()
    total_elapsed = time.time() - t_upload
    print(f"upload phase took {total_elapsed:.1f}s total "
          f"({len(all_files) / total_elapsed:.0f} files/s average)")

    train_count = len([
        n for n in vol.listdir("dataset_v3_50k/train")
        if n.path.startswith("dataset_v3_50k/train/heatmap_")
    ])
    print(f"volume now has {train_count} heatmap files under dataset_v3_50k/train")
    print("Done! Dataset is unzipped and ready.")


@app.local_entrypoint()
def main():
    run_unzip.remote()