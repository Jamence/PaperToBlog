from functools import partial
from loguru import logger

import glob, os, requests, time, json, tarfile, threading
import importlib
import time
import inspect
import re
import os
import base64
import shutil
import glob

pj = os.path.join
ARXIV_CACHE_DIR = "D:\2_Myself\project\PaperToBlog\arxiv_save_dir"


def zip_extract_member_new(self, member, targetpath, pwd):
    # 修复中文乱码的问题
    """Extract the ZipInfo object 'member' to a physical
        file on the path targetpath.
    """
    import zipfile
    if not isinstance(member, zipfile.ZipInfo):
        member = self.getinfo(member)

    # build the destination pathname, replacing
    # forward slashes to platform specific separators.
    arcname = member.filename.replace('/', os.path.sep)
    arcname = arcname.encode('cp437', errors='replace').decode('gbk', errors='replace')

    if os.path.altsep:
        arcname = arcname.replace(os.path.altsep, os.path.sep)
    # interpret absolute pathname as relative, remove drive letter or
    # UNC path, redundant separators, "." and ".." components.
    arcname = os.path.splitdrive(arcname)[1]
    invalid_path_parts = ('', os.path.curdir, os.path.pardir)
    arcname = os.path.sep.join(x for x in arcname.split(os.path.sep)
                                if x not in invalid_path_parts)
    if os.path.sep == '\\':
        # filter illegal characters on Windows
        arcname = self._sanitize_windows_name(arcname, os.path.sep)

    targetpath = os.path.join(targetpath, arcname)
    targetpath = os.path.normpath(targetpath)

    # Create all upper directories if necessary.
    upperdirs = os.path.dirname(targetpath)
    if upperdirs and not os.path.exists(upperdirs):
        os.makedirs(upperdirs)

    if member.is_dir():
        if not os.path.isdir(targetpath):
            os.mkdir(targetpath)
        return targetpath

    with self.open(member, pwd=pwd) as source, \
            open(targetpath, "wb") as target:
        shutil.copyfileobj(source, target)

    return targetpath

def extract_archive(file_path, dest_dir):
    import zipfile
    import tarfile
    import os

    # Get the file extension of the input file
    file_extension = os.path.splitext(file_path)[1]

    # Extract the archive based on its extension
    if file_extension == ".zip":
        with zipfile.ZipFile(file_path, "r") as zipobj:
            zipobj._extract_member = lambda a,b,c: zip_extract_member_new(zipobj, a,b,c)    # 修复中文乱码的问题
            zipobj.extractall(path=dest_dir)
            logger.info("Successfully extracted zip archive to {}".format(dest_dir))

    elif file_extension in [".tar", ".gz", ".bz2"]:
        try:
            with tarfile.open(file_path, "r:*") as tarobj:
                # 清理提取路径，移除任何不安全的元素
                for member in tarobj.getmembers():
                    member_path = os.path.normpath(member.name)
                    full_path = os.path.join(dest_dir, member_path)
                    full_path = os.path.abspath(full_path)
                    if not full_path.startswith(os.path.abspath(dest_dir) + os.sep):
                        raise Exception(f"Attempted Path Traversal in {member.name}")

                tarobj.extractall(path=dest_dir)
                logger.info("Successfully extracted tar archive to {}".format(dest_dir))
        except tarfile.ReadError as e:
            if file_extension == ".gz":
                # 一些特别奇葩的项目，是一个gz文件，里面不是tar，只有一个tex文件
                import gzip
                with gzip.open(file_path, 'rb') as f_in:
                    with open(os.path.join(dest_dir, 'main.tex'), 'wb') as f_out:
                        f_out.write(f_in.read())
            else:
                raise e

    # 第三方库，需要预先pip install rarfile
    # 此外，Windows上还需要安装winrar软件，配置其Path环境变量，如"C:\Program Files\WinRAR"才可以
    elif file_extension == ".rar":
        try:
            import rarfile

            with rarfile.RarFile(file_path) as rf:
                rf.extractall(path=dest_dir)
                logger.info("Successfully extracted rar archive to {}".format(dest_dir))
        except:
            logger.info("Rar format requires additional dependencies to install")
            return "\n\n解压失败! 需要安装pip install rarfile来解压rar文件。建议：使用zip压缩格式。"

    # 第三方库，需要预先pip install py7zr
    elif file_extension == ".7z":
        try:
            import py7zr

            with py7zr.SevenZipFile(file_path, mode="r") as f:
                f.extractall(path=dest_dir)
                logger.info("Successfully extracted 7z archive to {}".format(dest_dir))
        except:
            logger.info("7z format requires additional dependencies to install")
            return "\n\n解压失败! 需要安装pip install py7zr来解压7z文件"
    else:
        return ""
    return ""

#
#
# def arxiv_download(txt, allow_cache=True):
#
#     def is_float(s):
#         try:
#             float(s)
#             return True
#         except ValueError:
#             return False
#
#     if txt.startswith('https://arxiv.org/pdf/'):
#         arxiv_id = txt.split('/')[-1]   # 2402.14207v2.pdf
#         txt = arxiv_id.split('v')[0]  # 2402.14207
#
#     if ('.' in txt) and ('/' not in txt) and is_float(txt):  # is arxiv ID
#         txt = 'https://arxiv.org/abs/' + txt.strip()
#     if ('.' in txt) and ('/' not in txt) and is_float(txt[:10]):  # is arxiv ID
#         txt = 'https://arxiv.org/abs/' + txt[:10]
#
#     if not txt.startswith('https://arxiv.org'):
#         return txt, None    # 是本地文件，跳过下载
#
#     # <-------------- inspect format ------------->
#
#     url_ = txt  # https://arxiv.org/abs/1707.06690
#
#     if not txt.startswith('https://arxiv.org/abs/'):
#         msg = f"解析arxiv网址失败, 期望格式例如: https://arxiv.org/abs/1707.06690。实际得到格式: {url_}。"
#         return msg, None
#     # <-------------- set format ------------->
#     arxiv_id = url_.split('/abs/')[-1]
#     if 'v' in arxiv_id: arxiv_id = arxiv_id[:10]
#
#     extract_dst = pj(ARXIV_CACHE_DIR, arxiv_id, 'extract')
#     translation_dir = pj(ARXIV_CACHE_DIR, arxiv_id, 'e-print')
#     dst = pj(translation_dir, arxiv_id + '.tar')
#     os.makedirs(translation_dir, exist_ok=True)
#     # <-------------- download arxiv source file ------------->
#
#     def fix_url_and_download():
#         # for url_tar in [url_.replace('/abs/', '/e-print/'), url_.replace('/abs/', '/src/')]:
#         for url_tar in [url_.replace('/abs/', '/src/'), url_.replace('/abs/', '/e-print/')]:
#             proxies = get_conf('proxies')
#             r = requests.get(url_tar, proxies=proxies)
#             if r.status_code == 200:
#                 with open(dst, 'wb+') as f:
#                     f.write(r.content)
#                 return True
#         return False
#
#     if os.path.exists(dst) and allow_cache:
#         success = True
#     else:
#         success = fix_url_and_download()
#
#     if not success:
#         raise tarfile.ReadError(f"论文下载失败 {arxiv_id}")
#
#     # <-------------- extract file ------------->
#     from toolbox import extract_archive
#     try:
#         extract_archive(file_path=dst, dest_dir=extract_dst)
#     except tarfile.ReadError:
#         os.remove(dst)
#         raise tarfile.ReadError(f"论文下载失败")
#     return extract_dst, arxiv_id
if __name__ == "__main__" :
    extract_archive(file_path="D://2_Myself//project//PaperToBlog//read_paper//arXiv-2103.00020v1.tar.gz", dest_dir="./")