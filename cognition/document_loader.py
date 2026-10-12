"""
多格式文档加载器
================
支持 .txt / .md / .pdf / .docx / .html 五种格式的知识库文档解析。
返回统一格式: [{"source": "文件名", "content": "纯文本内容"}, ...]

依赖:
- .txt / .md: 无额外依赖
- .pdf: pip install pymupdf
- .docx: pip install python-docx
- .html: 无额外依赖（内置 html.parser）
"""

import re
from pathlib import Path
from html.parser import HTMLParser

from loguru import logger

DOCS_DIR = Path(__file__).resolve().parent / "docs"

# 支持的扩展名 → 解析器映射
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".html", ".htm"}


# ══════════════════════════════════════════════════════════
#  纯文本 / Markdown
# ══════════════════════════════════════════════════════════

def _load_text(fpath: Path) -> str:
    return fpath.read_text(encoding="utf-8")


# ══════════════════════════════════════════════════════════
#  PDF（需要 pymupdf）
# ══════════════════════════════════════════════════════════

def _load_pdf(fpath: Path) -> str:
    try:
        import fitz  # pymupdf
    except ImportError:
        logger.warning(f"pymupdf 未安装，跳过 PDF: {fpath.name}。"
                       f"安装: pip install pymupdf")
        return ""

    try:
        doc = fitz.open(str(fpath))
        pages = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages.append(text.strip())
        doc.close()
        content = "\n\n".join(pages)
        logger.info(f"PDF 解析完成: {fpath.name}，{len(pages)} 页，{len(content)} 字符")
        return content
    except Exception as e:
        logger.error(f"PDF 解析失败 {fpath.name}: {e}")
        return ""


# ══════════════════════════════════════════════════════════
#  Word .docx（需要 python-docx）
# ══════════════════════════════════════════════════════════

def _load_docx(fpath: Path) -> str:
    try:
        from docx import Document
    except ImportError:
        logger.warning(f"python-docx 未安装，跳过 Word: {fpath.name}。"
                       f"安装: pip install python-docx")
        return ""

    try:
        doc = Document(str(fpath))
        paragraphs = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                paragraphs.append(text)

        # 也提取表格内容
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    paragraphs.append(" | ".join(cells))

        content = "\n\n".join(paragraphs)
        logger.info(f"Word 解析完成: {fpath.name}，{len(paragraphs)} 段，{len(content)} 字符")
        return content
    except Exception as e:
        logger.error(f"Word 解析失败 {fpath.name}: {e}")
        return ""


# ══════════════════════════════════════════════════════════
#  HTML（内置 html.parser，无额外依赖）
# ══════════════════════════════════════════════════════════

class _HTMLTextExtractor(HTMLParser):
    """从 HTML 中提取纯文本，跳过 script/style 标签。"""

    def __init__(self):
        super().__init__()
        self._result = []
        self._skip = False
        self._skip_tags = {"script", "style", "head"}

    def handle_starttag(self, tag, attrs):
        if tag in self._skip_tags:
            self._skip = True
        # 块级标签加换行
        if tag in ("p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"):
            self._result.append("\n")

    def handle_endtag(self, tag):
        if tag in self._skip_tags:
            self._skip = False
        if tag in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "table"):
            self._result.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self._result.append(data)

    def get_text(self) -> str:
        raw = "".join(self._result)
        # 清理多余空行
        return re.sub(r"\n{3,}", "\n\n", raw).strip()


def _load_html(fpath: Path) -> str:
    try:
        html_content = fpath.read_text(encoding="utf-8")
        extractor = _HTMLTextExtractor()
        extractor.feed(html_content)
        text = extractor.get_text()
        logger.info(f"HTML 解析完成: {fpath.name}，{len(text)} 字符")
        return text
    except Exception as e:
        logger.error(f"HTML 解析失败 {fpath.name}: {e}")
        return ""


# ══════════════════════════════════════════════════════════
#  统一加载入口
# ══════════════════════════════════════════════════════════

_LOADERS = {
    ".txt": _load_text,
    ".md": _load_text,
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".html": _load_html,
    ".htm": _load_html,
}


def load_documents(docs_dir: Path = None) -> list[dict[str, str]]:
    """
    加载 docs_dir 下所有支持格式的文档。

    返回: [{"source": "文件名", "content": "纯文本内容"}, ...]
    """
    if docs_dir is None:
        docs_dir = DOCS_DIR

    docs: list[dict[str, str]] = []
    if not docs_dir.exists():
        logger.warning(f"文档目录不存在: {docs_dir}")
        return docs

    for fpath in sorted(docs_dir.iterdir()):
        if fpath.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if fpath.name.startswith("~") or fpath.name.startswith("."):
            continue

        loader = _LOADERS.get(fpath.suffix.lower())
        if loader is None:
            continue

        content = loader(fpath)
        if content.strip():
            docs.append({
                "source": fpath.name,
                "content": content,
            })
            logger.debug(f"已加载: {fpath.name} ({len(content)} 字符)")
        else:
            logger.warning(f"文档内容为空，已跳过: {fpath.name}")

    logger.info(f"共加载 {len(docs)} 个文档（目录: {docs_dir}）")
    return docs


# ══════════════════════════════════════════════════════════
#  快速测试
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    docs = load_documents()
    for d in docs:
        preview = d["content"][:100].replace("\n", " ")
        print(f"  {d['source']:30s}  {len(d['content']):>6d} 字符  {preview}...")
