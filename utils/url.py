# https://github.com/snap-stanford/ogb/blob/master/ogb/utils/url.py

import urllib.request as ur
import zipfile
import os
import os.path as osp
import errno
from tqdm import tqdm
import shutil

GBFACTOR = float(1 << 30)
CHUNK_SIZE = 1024 * 1024 * 1024 * 10  # 10 GB

def decide_download(url):
    d = ur.urlopen(url)
    size = int(d.info()["Content-Length"]) / GBFACTOR
    return input("Download %.3f GB processed data? (y/n)\n" % (size)).lower() == "y"


def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


def download_url(url, folder, log=True):
    r"""Downloads the content of an URL to a specific folder.
    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    filename = url.rpartition("/")[2]
    path = osp.join(folder, filename)

    if osp.exists(path) and osp.getsize(path) > 0:  # pragma: no cover
        if log:
            print("Using exist file", filename)
        return path

    if log:
        print("Downloading", url)

    makedirs(folder)
    data = ur.urlopen(url)

    size = int(data.info()["Content-Length"])

    chunk_size = 1024 * 1024
    num_iter = int(size / chunk_size) + 2

    downloaded_size = 0

    try:
        with open(path, "wb") as f:
            pbar = tqdm(range(num_iter))
            for i in pbar:
                chunk = data.read(chunk_size)
                downloaded_size += len(chunk)
                pbar.set_description("Downloaded {:.2f} GB".format(float(downloaded_size) / GBFACTOR))
                f.write(chunk)
    except:
        if os.path.exists(path):
            os.remove(path)
        raise RuntimeError("Stopped downloading due to interruption.")

    return path


def maybe_log(path, log=True):
    if log:
        print("Extracting", path)


#def extract_zip(path, folder, log=True):
#    r"""Extracts a zip archive to a specific folder.
#    Args:
#        path (string): The path to the tar archive.
#        folder (string): The folder.
#        log (bool, optional): If :obj:`False`, will not print anything to the
#            console. (default: :obj:`True`)
#    """
#    maybe_log(path, log)
#    with zipfile.ZipFile(path, "r") as f:
#        f.extractall(folder)


def extract_zip(path, folder, log=True):
    r"""Extracts a zip archive to a specific folder.
    Args:
        path (string): The path to the zip archive.
        folder (string): The folder.
        log (bool, optional): If :obj=`False`, will not print anything to the
            console. (default: :obj=`True`)
    """
    maybe_log(path, log)
    os.makedirs(folder, exist_ok=True)
    
    with zipfile.ZipFile(path, "r") as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc="Extracting"):
            try:
                extract_member(zip_ref, member, folder)
            except zipfile.error as e:
                print(f"Failed to extract {member.filename} due to {e}")
                
    print("Extraction completed.")

def extract_member(zip_ref, member, target_dir):
    """Extracts a single member from the zip file to the target directory."""
    target_path = os.path.join(target_dir, member.filename)
    
    if member.is_dir():
        os.makedirs(target_path, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with zip_ref.open(member) as source, open(target_path, 'wb') as target:
            shutil.copyfileobj(source, target, CHUNK_SIZE)