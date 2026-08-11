"""Compare the local crossval tarballs against the Zenodo deposit by CRC-32.

Reads CRCs from the zip central directory (no extraction) and streams each local
tarball to compute its CRC. Confirms that the artifacts converted for the HF
export are byte-identical to the ones published with the paper.
"""

import os
import sys
import zipfile
import zlib

ZIP_PATH = '/nfs/roberts/scratch/pi_skr2/mcn26/mpac_hf/mpac_model_artifacts.zip'
LOCAL_ROOT = '/nfs/roberts/project/pi_skr2/shared/boda_ensembl_models'


def local_crc(path, chunk=1 << 22):
    crc = 0
    with open(path, 'rb') as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            crc = zlib.crc32(block, crc)
    return crc & 0xFFFFFFFF


def main():
    with zipfile.ZipFile(ZIP_PATH) as zf:
        entries = [i for i in zf.infolist() if not i.is_dir()]
    zip_by_name = {}
    for info in entries:
        name = os.path.basename(info.filename)
        if not name or name.startswith('.'):
            continue
        zip_by_name.setdefault(name, []).append(info)

    print(f'zip entries: {len(entries)}, unique basenames: {len(zip_by_name)}')
    dupes = {k: len(v) for k, v in zip_by_name.items() if len(v) > 1}
    if dupes:
        print(f'warning: duplicate basenames in zip: {dupes}', file=sys.stderr)

    local = {}
    for fold in sorted(os.listdir(LOCAL_ROOT)):
        fold_dir = os.path.join(LOCAL_ROOT, fold)
        if not os.path.isdir(fold_dir):
            continue
        for name in sorted(os.listdir(fold_dir)):
            if name.endswith('.tar.gz'):
                local[name] = os.path.join(fold_dir, name)
    print(f'local tarballs: {len(local)}')

    matched = missing = mismatched = 0
    for name, path in sorted(local.items()):
        if name not in zip_by_name:
            print(f'MISSING FROM ZENODO: {name}')
            missing += 1
            continue
        want = zip_by_name[name][0].CRC
        got = local_crc(path)
        if want == got:
            matched += 1
        else:
            print(f'CRC MISMATCH: {name} zenodo={want:08x} local={got:08x}')
            mismatched += 1

    extra = sorted(set(zip_by_name) - set(local))
    print(f'\nmatched {matched}/{len(local)}, missing {missing}, mismatched {mismatched}')
    if extra:
        print(f'in Zenodo but not local ({len(extra)}): {extra[:10]}')

    sys.exit(1 if (missing or mismatched) else 0)


if __name__ == '__main__':
    main()
