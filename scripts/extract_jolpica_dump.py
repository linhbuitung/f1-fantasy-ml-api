from pathlib import Path
import re
import shutil
import zipfile

project_root = Path(__file__).resolve().parents[1]
dvc_file = project_root / "data" / "raw" / "jolpica-dump.zip.dvc"
zip_dest = project_root / "data" / "raw" / "jolpica-dump.zip"
extract_dir = project_root / "data" / "raw" / "jolpica-dump"

def _find_cache_path_from_dvc(dvc_path: Path) -> Path | None:
    txt = dvc_path.read_text()
    m = re.search(r"outs:\s*\n- md5:\s*([0-9a-fA-F]+)", txt)
    if not m:
        return None
    md5 = m.group(1)
    cache_path = project_root / ".dvc" / "cache" / "files" / "md5" / md5[:2] / md5[2:]
    return cache_path if cache_path.exists() else None

def main():
    if not dvc_file.exists():
        print("No DVC pointer found:", dvc_file)
        return

    # if explicit zip already present, use it
    if not zip_dest.exists():
        cache_path = _find_cache_path_from_dvc(dvc_file)
        if cache_path:
            # copy cache blob to readable zip file and extract from it
            shutil.copy2(cache_path, zip_dest)
            print("Copied from DVC cache:", cache_path, "->", zip_dest)
        else:
            print("No cache file found and no local zip present.")
            return

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_dest, "r") as zf:
        zf.extractall(path=extract_dir)
    print("Extracted to:", extract_dir)

if __name__ == "__main__":
    main()