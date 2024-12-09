from loguru import logger
import os, glob, shutil,re
import numpy as np
import platform
import pickle
import multiprocessing

# <-------------- step1.论文解压 ------------->
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
    print(file_extension)

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

                print("successful")
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

file_path="D://2_Myself//project//PaperToBlog//arxiv_save_dir//arXiv-2411.14405v2.tar.gz"
dest_dir="D://2_Myself//project//PaperToBlog//arxiv_save_dir//arXiv-2411.14405v2//"
print(file_path, dest_dir)
extract_archive(file_path=file_path, dest_dir=dest_dir)


# <-------------- step2. 寻找主tex文件------------->
txt, arxiv_id = "D://2_Myself//project//PaperToBlog//arxiv_save_dir//arXiv-2411.14405v2//", "2411.14405"
project_folder = txt
file_manifest = [f for f in glob.glob(f'{project_folder}/**/*.tex', recursive=True)]
mode = 'proofread'
def rm_comments(main_file):
    new_file_remove_comment_lines = []
    for l in main_file.splitlines():
        # 删除整行的空注释
        if l.lstrip().startswith("%"):
            pass
        else:
            new_file_remove_comment_lines.append(l)
    main_file = "\n".join(new_file_remove_comment_lines)
    # main_file = re.sub(r"\\include{(.*?)}", r"\\input{\1}", main_file)  # 将 \include 命令转换为 \input 命令
    main_file = re.sub(r"(?<!\\)%.*", "", main_file)  # 使用正则表达式查找半行注释, 并替换为空字符串
    return main_file
def find_main_tex_file(file_manifest, mode):
    """
    在多Tex文档中，寻找主文件，必须包含documentclass，返回找到的第一个。
    P.S. 但愿没人把latex模板放在里面传进来 (6.25 加入判定latex模板的代码)
    """
    canidates = []
    for texf in file_manifest:
        if os.path.basename(texf).startswith("merge"):
            continue
        with open(texf, "r", encoding="utf8", errors="ignore") as f:
            file_content = f.read()
        if r"\documentclass" in file_content:
            canidates.append(texf)
        else:
            continue

    if len(canidates) == 0:
        raise RuntimeError("无法找到一个主Tex文件（包含documentclass关键字）")
    elif len(canidates) == 1:
        return canidates[0]
    else:  # if len(canidates) >= 2 通过一些Latex模板中常见（但通常不会出现在正文）的单词，对不同latex源文件扣分，取评分最高者返回
        canidates_score = []
        # 给出一些判定模板文档的词作为扣分项
        unexpected_words = [
            "\\LaTeX",
            "manuscript",
            "Guidelines",
            "font",
            "citations",
            "rejected",
            "blind review",
            "reviewers",
        ]
        expected_words = ["\\input", "\\ref", "\\cite"]
        for texf in canidates:
            canidates_score.append(0)
            with open(texf, "r", encoding="utf8", errors="ignore") as f:
                file_content = f.read()
                file_content = rm_comments(file_content)
            for uw in unexpected_words:
                if uw in file_content:
                    canidates_score[-1] -= 1
            for uw in expected_words:
                if uw in file_content:
                    canidates_score[-1] += 1
        select = np.argmax(canidates_score)  # 取评分最高者返回
        return canidates[select]
maintex = find_main_tex_file(file_manifest, mode)
logger.info(f"maintex: {maintex}")


# <-------------- step3. tex文件合并------------->
pj = os.path.join
main_tex_basename = os.path.basename(maintex)
assert main_tex_basename.endswith('.tex')
main_tex_basename_bare = main_tex_basename[:-4]
may_exist_bbl = pj(project_folder, f'{main_tex_basename_bare}.bbl')
if os.path.exists(may_exist_bbl):
    shutil.copyfile(may_exist_bbl, pj(project_folder, f'merge.bbl'))
    shutil.copyfile(may_exist_bbl, pj(project_folder, f'merge_{mode}.bbl'))
    shutil.copyfile(may_exist_bbl, pj(project_folder, f'merge_diff.bbl'))

insert_missing_abs_str = r"""
\begin{abstract}
The GPT-Academic program cannot find abstract section in this paper.
\end{abstract}
"""
def find_tex_file_ignore_case(fp):
    dir_name = os.path.dirname(fp)
    base_name = os.path.basename(fp)
    # 如果输入的文件路径是正确的
    if os.path.isfile(pj(dir_name, base_name)):
        return pj(dir_name, base_name)
    # 如果不正确，试着加上.tex后缀试试
    if not base_name.endswith(".tex"):
        base_name += ".tex"
    if os.path.isfile(pj(dir_name, base_name)):
        return pj(dir_name, base_name)
    # 如果还找不到，解除大小写限制，再试一次
    import glob

    for f in glob.glob(dir_name + "/*.tex"):
        base_name_s = os.path.basename(fp)
        base_name_f = os.path.basename(f)
        if base_name_s.lower() == base_name_f.lower():
            return f
        # 试着加上.tex后缀试试
        if not base_name_s.endswith(".tex"):
            base_name_s += ".tex"
        if base_name_s.lower() == base_name_f.lower():
            return f
    return None
def insert_abstract(tex_content):
    if "\\maketitle" in tex_content:
        # find the position of "\maketitle"
        find_index = tex_content.index("\\maketitle")
        # find the nearest ending line
        end_line_index = tex_content.find("\n", find_index)
        # insert "abs_str" on the next line
        modified_tex = (
            tex_content[: end_line_index + 1]
            + "\n\n"
            + insert_missing_abs_str
            + "\n\n"
            + tex_content[end_line_index + 1 :]
        )
        return modified_tex
    elif r"\begin{document}" in tex_content:
        # find the position of "\maketitle"
        find_index = tex_content.index(r"\begin{document}")
        # find the nearest ending line
        end_line_index = tex_content.find("\n", find_index)
        # insert "abs_str" on the next line
        modified_tex = (
            tex_content[: end_line_index + 1]
            + "\n\n"
            + insert_missing_abs_str
            + "\n\n"
            + tex_content[end_line_index + 1 :]
        )
        return modified_tex
    else:
        return tex_content
def merge_tex_files_(project_foler, main_file, mode):
    """
    Merge Tex project recrusively
    """
    main_file = rm_comments(main_file)
    for s in reversed([q for q in re.finditer(r"\\input\{(.*?)\}", main_file, re.M)]):
        f = s.group(1)
        fp = os.path.join(project_foler, f)
        fp_ = find_tex_file_ignore_case(fp)
        if fp_:
            try:
                with open(fp_, "r", encoding="utf-8", errors="replace") as fx:
                    c = fx.read()
            except:
                c = f"\n\nWarning from GPT-Academic: LaTex source file is missing!\n\n"
        else:
            raise RuntimeError(f"找不到{fp}，Tex源文件缺失！")
        c = merge_tex_files_(project_foler, c, mode)
        main_file = main_file[: s.span()[0]] + c + main_file[s.span()[1] :]
    return main_file
def merge_tex_files(project_foler, main_file, mode):
    """
    Merge Tex project recrusively
    P.S. 顺便把CTEX塞进去以支持中文
    P.S. 顺便把Latex的注释去除
    """
    main_file = merge_tex_files_(project_foler, main_file, mode)
    main_file = rm_comments(main_file)

    if mode == "translate_zh":
        # find paper documentclass
        pattern = re.compile(r"\\documentclass.*\n")
        match = pattern.search(main_file)
        assert match is not None, "Cannot find documentclass statement!"
        position = match.end()
        add_ctex = "\\usepackage{ctex}\n"
        add_url = "\\usepackage{url}\n" if "{url}" not in main_file else ""
        main_file = main_file[:position] + add_ctex + add_url + main_file[position:]
        # fontset=windows
        import platform

        main_file = re.sub(
            r"\\documentclass\[(.*?)\]{(.*?)}",
            r"\\documentclass[\1,fontset=windows,UTF8]{\2}",
            main_file,
        )
        main_file = re.sub(
            r"\\documentclass{(.*?)}",
            r"\\documentclass[fontset=windows,UTF8]{\1}",
            main_file,
        )
        # find paper abstract
        pattern_opt1 = re.compile(r"\\begin\{abstract\}.*\n")
        pattern_opt2 = re.compile(r"\\abstract\{(.*?)\}", flags=re.DOTALL)
        match_opt1 = pattern_opt1.search(main_file)
        match_opt2 = pattern_opt2.search(main_file)
        if (match_opt1 is None) and (match_opt2 is None):
            # "Cannot find paper abstract section!"
            main_file = insert_abstract(main_file)
        match_opt1 = pattern_opt1.search(main_file)
        match_opt2 = pattern_opt2.search(main_file)
        assert (match_opt1 is not None) or (
            match_opt2 is not None
        ), "Cannot find paper abstract section!"
    return main_file

with open(maintex, 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()
    merged_content = merge_tex_files(project_folder, content, mode)

with open(project_folder + '/merge.tex', 'w', encoding='utf-8', errors='replace') as f:
    f.write(merged_content)
logger.info(f"merge.tex写入完成")



# <-------------- step4. tex分割------------->
PRESERVE = 0
TRANSFORM = 1
def find_title_and_abs(main_file):
    def extract_abstract_1(text):
        pattern = r"\\abstract\{(.*?)\}"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        else:
            return None

    def extract_abstract_2(text):
        pattern = r"\\begin\{abstract\}(.*?)\\end\{abstract\}"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        else:
            return None

    def extract_title(string):
        pattern = r"\\title\{(.*?)\}"
        match = re.search(pattern, string, re.DOTALL)

        if match:
            return match.group(1)
        else:
            return None

    abstract = extract_abstract_1(main_file)
    if abstract is None:
        abstract = extract_abstract_2(main_file)
    title = extract_title(main_file)
    return title, abstract
def mod_inbraket(match):
    """
    为啥chatgpt会把cite里面的逗号换成中文逗号呀
    """
    # get the matched string
    cmd = match.group(1)
    str_to_modify = match.group(2)
    # modify the matched string
    str_to_modify = str_to_modify.replace("：", ":")  # 前面是中文冒号，后面是英文冒号
    str_to_modify = str_to_modify.replace("，", ",")  # 前面是中文逗号，后面是英文逗号
    # str_to_modify = 'BOOM'
    return "\\" + cmd + "{" + str_to_modify + "}"
def fix_content(final_tex, node_string):
    """
    Fix common GPT errors to increase success rate
    """
    final_tex = re.sub(r"(?<!\\)%", "\\%", final_tex)
    final_tex = re.sub(r"\\([a-z]{2,10})\ \{", r"\\\1{", string=final_tex)
    final_tex = re.sub(r"\\\ ([a-z]{2,10})\{", r"\\\1{", string=final_tex)
    final_tex = re.sub(r"\\([a-z]{2,10})\{([^\}]*?)\}", mod_inbraket, string=final_tex)

    if "Traceback" in final_tex and "[Local Message]" in final_tex:
        final_tex = node_string  # 出问题了，还原原文
    if node_string.count("\\begin") != final_tex.count("\\begin"):
        final_tex = node_string  # 出问题了，还原原文
    if node_string.count("\_") > 0 and node_string.count("\_") > final_tex.count("\_"):
        # walk and replace any _ without \
        final_tex = re.sub(r"(?<!\\)_", "\\_", final_tex)

    def compute_brace_level(string):
        # this function count the number of { and }
        brace_level = 0
        for c in string:
            if c == "{":
                brace_level += 1
            elif c == "}":
                brace_level -= 1
        return brace_level

    def join_most(tex_t, tex_o):
        # this function join translated string and original string when something goes wrong
        p_t = 0
        p_o = 0

        def find_next(string, chars, begin):
            p = begin
            while p < len(string):
                if string[p] in chars:
                    return p, string[p]
                p += 1
            return None, None

        while True:
            res1, char = find_next(tex_o, ["{", "}"], p_o)
            if res1 is None:
                break
            res2, char = find_next(tex_t, [char], p_t)
            if res2 is None:
                break
            p_o = res1 + 1
            p_t = res2 + 1
        return tex_t[:p_t] + tex_o[p_o:]

    if compute_brace_level(final_tex) != compute_brace_level(node_string):
        # 出问题了，还原部分原文，保证括号正确
        final_tex = join_most(final_tex, node_string)
    return final_tex
def set_forbidden_text(text, mask, pattern, flags=0):
    """
    Add a preserve text area in this paper
    e.g. with pattern = r"\\begin\{algorithm\}(.*?)\\end\{algorithm\}"
    you can mask out (mask = PRESERVE so that text become untouchable for GPT)
    everything between "\begin{equation}" and "\end{equation}"
    """
    if isinstance(pattern, list):
        pattern = "|".join(pattern)
    pattern_compile = re.compile(pattern, flags)
    for res in pattern_compile.finditer(text):
        mask[res.span()[0] : res.span()[1]] = PRESERVE
    return text, mask
def set_forbidden_text_begin_end(text, mask, pattern, flags=0, limit_n_lines=42):
    """
    Find all \begin{} ... \end{} text block that with less than limit_n_lines lines.
    Add it to preserve area
    """
    pattern_compile = re.compile(pattern, flags)

    def search_with_line_limit(text, mask):
        for res in pattern_compile.finditer(text):
            cmd = res.group(1)  # begin{what}
            this = res.group(2)  # content between begin and end
            this_mask = mask[res.regs[2][0] : res.regs[2][1]]
            white_list = [
                "document",
                "abstract",
                "lemma",
                "definition",
                "sproof",
                "em",
                "emph",
                "textit",
                "textbf",
                "itemize",
                "enumerate",
            ]
            if (cmd in white_list) or this.count(
                "\n"
            ) >= limit_n_lines:  # use a magical number 42
                this, this_mask = search_with_line_limit(this, this_mask)
                mask[res.regs[2][0] : res.regs[2][1]] = this_mask
            else:
                mask[res.regs[0][0] : res.regs[0][1]] = PRESERVE
        return text, mask

    return search_with_line_limit(text, mask)
def set_forbidden_text_careful_brace(text, mask, pattern, flags=0):
    """
    Add a preserve text area in this paper (text become untouchable for GPT).
    count the number of the braces so as to catch compelete text area.
    e.g.
    \caption{blablablablabla\texbf{blablabla}blablabla.}
    """
    pattern_compile = re.compile(pattern, flags)
    for res in pattern_compile.finditer(text):
        brace_level = -1
        p = begin = end = res.regs[0][0]
        for _ in range(1024 * 16):
            if text[p] == "}" and brace_level == 0:
                break
            elif text[p] == "}":
                brace_level -= 1
            elif text[p] == "{":
                brace_level += 1
            p += 1
        end = p + 1
        mask[begin:end] = PRESERVE
    return text, mask
def reverse_forbidden_text_careful_brace(
    text, mask, pattern, flags=0, forbid_wrapper=True
):
    """
    Move area out of preserve area (make text editable for GPT)
    count the number of the braces so as to catch compelete text area.
    e.g.
    \caption{blablablablabla\texbf{blablabla}blablabla.}
    """
    pattern_compile = re.compile(pattern, flags)
    for res in pattern_compile.finditer(text):
        brace_level = 0
        p = begin = end = res.regs[1][0]
        for _ in range(1024 * 16):
            if text[p] == "}" and brace_level == 0:
                break
            elif text[p] == "}":
                brace_level -= 1
            elif text[p] == "{":
                brace_level += 1
            p += 1
        end = p
        mask[begin:end] = TRANSFORM
        if forbid_wrapper:
            mask[res.regs[0][0] : begin] = PRESERVE
            mask[end : res.regs[0][1]] = PRESERVE
    return text, mask
def reverse_forbidden_text(text, mask, pattern, flags=0, forbid_wrapper=True):
    """
    Move area out of preserve area (make text editable for GPT)
    count the number of the braces so as to catch compelete text area.
    e.g.
    \begin{abstract} blablablablablabla. \end{abstract}
    """
    if isinstance(pattern, list):
        pattern = "|".join(pattern)
    pattern_compile = re.compile(pattern, flags)
    for res in pattern_compile.finditer(text):
        if not forbid_wrapper:
            mask[res.span()[0] : res.span()[1]] = TRANSFORM
        else:
            mask[res.regs[0][0] : res.regs[1][0]] = PRESERVE  # '\\begin{abstract}'
            mask[res.regs[1][0] : res.regs[1][1]] = TRANSFORM  # abstract
            mask[res.regs[1][1] : res.regs[0][1]] = PRESERVE  # abstract
    return text, mask
class LinkedListNode:
    """
    Linked List Node
    """

    def __init__(self, string, preserve=True) -> None:
        self.string = string
        self.preserve = preserve
        self.next = None
        self.range = None
        # self.begin_line = 0
        # self.begin_char = 0
def convert_to_linklist(text, mask):
    root = LinkedListNode("", preserve=True)
    current_node = root
    for c, m, i in zip(text, mask, range(len(text))):
        if (m == PRESERVE and current_node.preserve) or (
            m == TRANSFORM and not current_node.preserve
        ):
            # add
            current_node.string += c
        else:
            current_node.next = LinkedListNode(c, preserve=(m == PRESERVE))
            current_node = current_node.next
    return root
def post_process(root):
    # 修复括号
    node = root
    while True:
        string = node.string
        if node.preserve:
            node = node.next
            if node is None:
                break
            continue

        def break_check(string):
            str_stack = [""]  # (lv, index)
            for i, c in enumerate(string):
                if c == "{":
                    str_stack.append("{")
                elif c == "}":
                    if len(str_stack) == 1:
                        logger.warning("fixing brace error")
                        return i
                    str_stack.pop(-1)
                else:
                    str_stack[-1] += c
            return -1

        bp = break_check(string)

        if bp == -1:
            pass
        elif bp == 0:
            node.string = string[:1]
            q = LinkedListNode(string[1:], False)
            q.next = node.next
            node.next = q
        else:
            node.string = string[:bp]
            q = LinkedListNode(string[bp:], False)
            q.next = node.next
            node.next = q

        node = node.next
        if node is None:
            break

    # 屏蔽空行和太短的句子
    node = root
    while True:
        if len(node.string.strip("\n").strip("")) == 0:
            node.preserve = True
        if len(node.string.strip("\n").strip("")) < 42:
            node.preserve = True
        node = node.next
        if node is None:
            break
    node = root
    while True:
        if node.next and node.preserve and node.next.preserve:
            node.string += node.next.string
            node.next = node.next.next
        node = node.next
        if node is None:
            break

    # 将前后断行符脱离
    node = root
    prev_node = None
    while True:
        if not node.preserve:
            lstriped_ = node.string.lstrip().lstrip("\n")
            if (
                (prev_node is not None)
                and (prev_node.preserve)
                and (len(lstriped_) != len(node.string))
            ):
                prev_node.string += node.string[: -len(lstriped_)]
                node.string = lstriped_
            rstriped_ = node.string.rstrip().rstrip("\n")
            if (
                (node.next is not None)
                and (node.next.preserve)
                and (len(rstriped_) != len(node.string))
            ):
                node.next.string = node.string[len(rstriped_) :] + node.next.string
                node.string = rstriped_
        # =-=-=
        prev_node = node
        node = node.next
        if node is None:
            break

    # 标注节点的行数范围
    node = root
    n_line = 0
    expansion = 2
    while True:
        n_l = node.string.count("\n")
        node.range = [n_line - expansion, n_line + n_l + expansion]  # 失败时，扭转的范围
        n_line = n_line + n_l
        node = node.next
        if node is None:
            break
    return root
def split_subprocess(txt, project_folder, return_dict, opts):
    """
    break down latex file to a linked list,
    each node use a preserve flag to indicate whether it should
    be proccessed by GPT.
    """
    text = txt
    mask = np.zeros(len(txt), dtype=np.uint8) + TRANSFORM

    # 吸收title与作者以上的部分
    text, mask = set_forbidden_text(text, mask, r"^(.*?)\\maketitle", re.DOTALL)
    text, mask = set_forbidden_text(text, mask, r"^(.*?)\\begin{document}", re.DOTALL)
    # 吸收iffalse注释
    text, mask = set_forbidden_text(text, mask, r"\\iffalse(.*?)\\fi", re.DOTALL)
    # 吸收在42行以内的begin-end组合
    text, mask = set_forbidden_text_begin_end(text, mask, r"\\begin\{([a-z\*]*)\}(.*?)\\end\{\1\}", re.DOTALL, limit_n_lines=42)
    # 吸收匿名公式
    text, mask = set_forbidden_text(text, mask, [ r"\$\$([^$]+)\$\$",  r"\\\[.*?\\\]" ], re.DOTALL)
    # 吸收其他杂项
    text, mask = set_forbidden_text(text, mask, [ r"\\section\{(.*?)\}", r"\\section\*\{(.*?)\}", r"\\subsection\{(.*?)\}", r"\\subsubsection\{(.*?)\}" ])
    text, mask = set_forbidden_text(text, mask, [ r"\\bibliography\{(.*?)\}", r"\\bibliographystyle\{(.*?)\}" ])
    text, mask = set_forbidden_text(text, mask, r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", re.DOTALL)
    text, mask = set_forbidden_text(text, mask, r"\\begin\{lstlisting\}(.*?)\\end\{lstlisting\}", re.DOTALL)
    text, mask = set_forbidden_text(text, mask, r"\\begin\{wraptable\}(.*?)\\end\{wraptable\}", re.DOTALL)
    text, mask = set_forbidden_text(text, mask, r"\\begin\{algorithm\}(.*?)\\end\{algorithm\}", re.DOTALL)
    text, mask = set_forbidden_text(text, mask, [r"\\begin\{wrapfigure\}(.*?)\\end\{wrapfigure\}", r"\\begin\{wrapfigure\*\}(.*?)\\end\{wrapfigure\*\}"], re.DOTALL)
    text, mask = set_forbidden_text(text, mask, [r"\\begin\{figure\}(.*?)\\end\{figure\}", r"\\begin\{figure\*\}(.*?)\\end\{figure\*\}"], re.DOTALL)
    text, mask = set_forbidden_text(text, mask, [r"\\begin\{multline\}(.*?)\\end\{multline\}", r"\\begin\{multline\*\}(.*?)\\end\{multline\*\}"], re.DOTALL)
    text, mask = set_forbidden_text(text, mask, [r"\\begin\{table\}(.*?)\\end\{table\}", r"\\begin\{table\*\}(.*?)\\end\{table\*\}"], re.DOTALL)
    text, mask = set_forbidden_text(text, mask, [r"\\begin\{minipage\}(.*?)\\end\{minipage\}", r"\\begin\{minipage\*\}(.*?)\\end\{minipage\*\}"], re.DOTALL)
    text, mask = set_forbidden_text(text, mask, [r"\\begin\{align\*\}(.*?)\\end\{align\*\}", r"\\begin\{align\}(.*?)\\end\{align\}"], re.DOTALL)
    text, mask = set_forbidden_text(text, mask, [r"\\begin\{equation\}(.*?)\\end\{equation\}", r"\\begin\{equation\*\}(.*?)\\end\{equation\*\}"], re.DOTALL)
    text, mask = set_forbidden_text(text, mask, [r"\\includepdf\[(.*?)\]\{(.*?)\}", r"\\clearpage", r"\\newpage", r"\\appendix", r"\\tableofcontents", r"\\include\{(.*?)\}"])
    text, mask = set_forbidden_text(text, mask, [r"\\vspace\{(.*?)\}", r"\\hspace\{(.*?)\}", r"\\label\{(.*?)\}", r"\\begin\{(.*?)\}", r"\\end\{(.*?)\}", r"\\item "])
    text, mask = set_forbidden_text_careful_brace(text, mask, r"\\hl\{(.*?)\}", re.DOTALL)
    # reverse 操作必须放在最后
    text, mask = reverse_forbidden_text_careful_brace(text, mask, r"\\caption\{(.*?)\}", re.DOTALL, forbid_wrapper=True)
    text, mask = reverse_forbidden_text_careful_brace(text, mask, r"\\abstract\{(.*?)\}", re.DOTALL, forbid_wrapper=True)
    text, mask = reverse_forbidden_text(text, mask, r"\\begin\{abstract\}(.*?)\\end\{abstract\}", re.DOTALL, forbid_wrapper=True)
    root = convert_to_linklist(text, mask)

    # 最后一步处理，增强稳健性
    root = post_process(root)

    # 输出html调试文件，用红色标注处保留区（PRESERVE），用黑色标注转换区（TRANSFORM）
    with open(pj(project_folder, 'debug_log.html'), 'w', encoding='utf8') as f:
        segment_parts_for_gpt = []
        nodes = []
        node = root
        while True:
            nodes.append(node)
            show_html = node.string.replace('\n','<br/>')
            if not node.preserve:
                segment_parts_for_gpt.append(node.string)
                f.write(f'<p style="color:black;">#{node.range}{show_html}#</p>')
            else:
                f.write(f'<p style="color:red;">{show_html}</p>')
            node = node.next
            if node is None: break

    for n in nodes: n.next = None   # break
    return_dict['nodes'] = nodes
    return_dict['segment_parts_for_gpt'] = segment_parts_for_gpt
    return return_dict
class LatexPaperSplit():
    """
    break down latex file to a linked list,
    each node use a preserve flag to indicate whether it should
    be proccessed by GPT.
    """
    def __init__(self) -> None:
        self.nodes = None
        self.msg = "*{\\scriptsize\\textbf{警告：该PDF由GPT-Academic开源项目调用大语言模型+Latex翻译插件一键生成，" + \
            "版权归原文作者所有。翻译内容可靠性无保障，请仔细鉴别并以原文为准。" + \
            "项目Github地址 \\url{https://github.com/binary-husky/gpt_academic/}。"
        # 请您不要删除或修改这行警告，除非您是论文的原作者（如果您是论文原作者，欢迎加REAME中的QQ联系开发者）
        self.msg_declare = "为了防止大语言模型的意外谬误产生扩散影响，禁止移除或修改此警告。}}\\\\"
        self.title = "unknown"
        self.abstract = "unknown"

    def read_title_and_abstract(self, txt):
        try:
            title, abstract = find_title_and_abs(txt)
            if title is not None:
                self.title = title.replace('\n', ' ').replace('\\\\', ' ').replace('  ', '').replace('  ', '')
            if abstract is not None:
                self.abstract = abstract.replace('\n', ' ').replace('\\\\', ' ').replace('  ', '').replace('  ', '')
        except:
            pass

    def merge_result(self, arr, mode, msg, buggy_lines=[], buggy_line_surgery_n_lines=10):
        """
        Merge the result after the GPT process completed
        """
        result_string = ""
        node_cnt = 0
        line_cnt = 0

        for node in self.nodes:
            if node.preserve:
                line_cnt += node.string.count('\n')
                result_string += node.string
            else:
                translated_txt = fix_content(arr[node_cnt], node.string)
                begin_line = line_cnt
                end_line = line_cnt + translated_txt.count('\n')

                # reverse translation if any error
                if any([begin_line-buggy_line_surgery_n_lines <= b_line <= end_line+buggy_line_surgery_n_lines for b_line in buggy_lines]):
                    translated_txt = node.string

                result_string += translated_txt
                node_cnt += 1
                line_cnt += translated_txt.count('\n')

        if mode == 'translate_zh':
            pattern = re.compile(r'\\begin\{abstract\}.*\n')
            match = pattern.search(result_string)
            if not match:
                # match \abstract{xxxx}
                pattern_compile = re.compile(r"\\abstract\{(.*?)\}", flags=re.DOTALL)
                match = pattern_compile.search(result_string)
                position = match.regs[1][0]
            else:
                # match \begin{abstract}xxxx\end{abstract}
                position = match.end()
            result_string = result_string[:position] + self.msg + msg + self.msg_declare + result_string[position:]
        return result_string


    def split(self, txt, project_folder, opts):
        """
        break down latex file to a linked list,
        each node use a preserve flag to indicate whether it should
        be proccessed by GPT.
        P.S. use multiprocessing to avoid timeout error
        """
        import multiprocessing
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        p = multiprocessing.Process(
            target=split_subprocess,
            args=(txt, project_folder, return_dict, opts))
        p.start()
        p.join()
        p.close()
        self.nodes = return_dict['nodes']
        self.sp = return_dict['segment_parts_for_gpt']
        return self.sp
def maintain_storage(remain_txt_to_cut, remain_txt_to_cut_storage):
    """ 为了加速计算，我们采样一个特殊的手段。当 remain_txt_to_cut > `_max` 时， 我们把 _max 后的文字转存至 remain_txt_to_cut_storage
    当 remain_txt_to_cut < `_min` 时，我们再把 remain_txt_to_cut_storage 中的部分文字取出
    """
    _min = int(5e4)
    _max = int(1e5)
    # print(len(remain_txt_to_cut), len(remain_txt_to_cut_storage))
    if len(remain_txt_to_cut) < _min and len(remain_txt_to_cut_storage) > 0:
        remain_txt_to_cut = remain_txt_to_cut + remain_txt_to_cut_storage
        remain_txt_to_cut_storage = ""
    if len(remain_txt_to_cut) > _max:
        remain_txt_to_cut_storage = remain_txt_to_cut[_max:] + remain_txt_to_cut_storage
        remain_txt_to_cut = remain_txt_to_cut[:_max]
    return remain_txt_to_cut, remain_txt_to_cut_storage
def force_breakdown(txt, limit, get_token_fn):
    """ 当无法用标点、空行分割时，我们用最暴力的方法切割
    """
    for i in reversed(range(len(txt))):
        if get_token_fn(txt[:i]) < limit:
            return txt[:i], txt[i:]
    return "Tiktoken未知错误", "Tiktoken未知错误"
def cut(limit, get_token_fn, txt_tocut, must_break_at_empty_line, break_anyway=False):
    """ 文本切分
    """
    res = []
    total_len = len(txt_tocut)
    fin_len = 0
    remain_txt_to_cut = txt_tocut
    remain_txt_to_cut_storage = ""
    # 为了加速计算，我们采样一个特殊的手段。当 remain_txt_to_cut > `_max` 时， 我们把 _max 后的文字转存至 remain_txt_to_cut_storage
    remain_txt_to_cut, remain_txt_to_cut_storage = maintain_storage(remain_txt_to_cut, remain_txt_to_cut_storage)

    while True:
        if get_token_fn(remain_txt_to_cut) <= limit:
            # 如果剩余文本的token数小于限制，那么就不用切了
            res.append(remain_txt_to_cut); fin_len+=len(remain_txt_to_cut)
            break
        else:
            # 如果剩余文本的token数大于限制，那么就切
            lines = remain_txt_to_cut.split('\n')

            # 估计一个切分点
            estimated_line_cut = limit / get_token_fn(remain_txt_to_cut) * len(lines)
            estimated_line_cut = int(estimated_line_cut)

            # 开始查找合适切分点的偏移（cnt）
            cnt = 0
            for cnt in reversed(range(estimated_line_cut)):
                if must_break_at_empty_line:
                    # 首先尝试用双空行（\n\n）作为切分点
                    if lines[cnt] != "":
                        continue
                prev = "\n".join(lines[:cnt])
                post = "\n".join(lines[cnt:])
                if get_token_fn(prev) < limit:
                    break

            if cnt == 0:
                # 如果没有找到合适的切分点
                if break_anyway:
                    # 是否允许暴力切分
                    prev, post = force_breakdown(remain_txt_to_cut, limit, get_token_fn)
                else:
                    # 不允许直接报错
                    raise RuntimeError(f"存在一行极长的文本！{remain_txt_to_cut}")

            # 追加列表
            res.append(prev); fin_len+=len(prev)
            # 准备下一次迭代
            remain_txt_to_cut = post
            remain_txt_to_cut, remain_txt_to_cut_storage = maintain_storage(remain_txt_to_cut, remain_txt_to_cut_storage)
            process = fin_len/total_len
            logger.info(f'正在文本切分 {int(process*100)}%')
            if len(remain_txt_to_cut.strip()) == 0:
                break
    return res
def breakdown_text_to_satisfy_token_limit_(txt, limit, llm_model="gpt-3.5-turbo"):
    """ 使用多种方式尝试切分文本，以满足 token 限制
    """
    # from request_llms.bridge_all import model_info
    # enc = model_info[llm_model]['tokenizer']
    # def get_token_fn(txt): return len(enc.encode(txt, disallowed_special=()))
    def get_token_fn(txt):
        return len(txt)
    try:
        # 第1次尝试，将双空行（\n\n）作为切分点
        return cut(limit, get_token_fn, txt, must_break_at_empty_line=True)
    except RuntimeError:
        try:
            # 第2次尝试，将单空行（\n）作为切分点
            return cut(limit, get_token_fn, txt, must_break_at_empty_line=False)
        except RuntimeError:
            try:
                # 第3次尝试，将英文句号（.）作为切分点
                res = cut(limit, get_token_fn, txt.replace('.', '。\n'), must_break_at_empty_line=False) # 这个中文的句号是故意的，作为一个标识而存在
                return [r.replace('。\n', '.') for r in res]
            except RuntimeError as e:
                try:
                    # 第4次尝试，将中文句号（。）作为切分点
                    res = cut(limit, get_token_fn, txt.replace('。', '。。\n'), must_break_at_empty_line=False)
                    return [r.replace('。。\n', '。') for r in res]
                except RuntimeError as e:
                    # 第5次尝试，没办法了，随便切一下吧
                    return cut(limit, get_token_fn, txt, must_break_at_empty_line=False, break_anyway=True)

def run_in_subprocess_wrapper_func(v_args):
    func, args, kwargs, return_dict, exception_dict = pickle.loads(v_args)
    import sys
    try:
        result = func(*args, **kwargs)
        return_dict['result'] = result
    except Exception as e:
        exc_info = sys.exc_info()
        exception_dict['exception'] = exc_info
def run_in_subprocess_with_timeout(func, timeout=60):
    if platform.system() == 'Linux':
        def wrapper(*args, **kwargs):
            return_dict = multiprocessing.Manager().dict()
            exception_dict = multiprocessing.Manager().dict()
            v_args = pickle.dumps((func, args, kwargs, return_dict, exception_dict))
            process = multiprocessing.Process(target=run_in_subprocess_wrapper_func, args=(v_args,))
            process.start()
            process.join(timeout)
            if process.is_alive():
                process.terminate()
                raise TimeoutError(f'功能单元{str(func)}未能在规定时间内完成任务')
            process.close()
            if 'exception' in exception_dict:
                # ooops, the subprocess ran into an exception
                exc_info = exception_dict['exception']
                raise exc_info[1].with_traceback(exc_info[2])
            if 'result' in return_dict.keys():
                # If the subprocess ran successfully, return the result
                return return_dict['result']
        return wrapper
    else:
        return func
breakdown_text_to_satisfy_token_limit = run_in_subprocess_with_timeout(breakdown_text_to_satisfy_token_limit_, timeout=60)
class LatexPaperFileGroup():
    """
    use tokenizer to break down text according to max_token_limit
    """
    def __init__(self):
        self.file_paths = []
        self.file_contents = []
        self.sp_file_contents = []
        self.sp_file_index = []
        self.sp_file_tag = []
        # count_token
        # from request_llms.bridge_all import model_info
        # enc = model_info["gpt-3.5-turbo"]['tokenizer']
        def get_token_num(txt): return len(txt)
        self.get_token_num = get_token_num

    def run_file_split(self, max_token_limit=1900):
        """
        use tokenizer to break down text according to max_token_limit
        """
        for index, file_content in enumerate(self.file_contents):
            if self.get_token_num(file_content) < max_token_limit:
                self.sp_file_contents.append(file_content)
                self.sp_file_index.append(index)
                self.sp_file_tag.append(self.file_paths[index])
            else:
                # from crazy_functions.pdf_fns.breakdown_txt import breakdown_text_to_satisfy_token_limit
                segments = breakdown_text_to_satisfy_token_limit(file_content, max_token_limit)
                for j, segment in enumerate(segments):
                    self.sp_file_contents.append(segment)
                    self.sp_file_index.append(index)
                    self.sp_file_tag.append(self.file_paths[index] + f".part-{j}.tex")

    def merge_result(self):
        self.file_result = ["" for _ in range(len(self.file_paths))]
        for r, k in zip(self.sp_file_result, self.sp_file_index):
            self.file_result[k] += r

    def write_result(self):
        manifest = []
        for path, res in zip(self.file_paths, self.file_result):
            with open(path + '.polish.tex', 'w', encoding='utf8') as f:
                manifest.append(path + '.polish.tex')
                f.write(res)
        return manifest

if __name__ == '__main__' :
    lps = LatexPaperSplit()
    logger.info(f"正在处理 LatexPaperSplit")
    lps.read_title_and_abstract(merged_content)
    opts=[]
    res = lps.split(merged_content, project_folder, opts)  # 消耗时间的函数
    #  <-------- 拆分过长的latex片段 ---------->
    pfg = LatexPaperFileGroup()
    for index, r in enumerate(res):
        pfg.file_paths.append('segment-' + str(index))
        pfg.file_contents.append(r)

    pfg.run_file_split(max_token_limit=1024)
    n_split = len(pfg.sp_file_contents)
    print()