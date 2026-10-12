"""
文档去重清洗脚本 v2
==================
按关键词检测并移除source文档中与target文档重复的bullet point/段落。
用法: python dedup_docs.py

策略:
- 定义需要去重的关键词（这些话题只在target中保留）
- 在source文档中逐行扫描，命中任一关键词即移除该行
- 保留首次出现的上下文引用（如布偶猫section提到化毛膏是合理的品种描述）
"""

from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent / "cognition" / "docs"


def dedup_by_keywords(
    source_path: Path,
    target_path: Path,
    keywords: list[str],
    context_lines_to_skip: int = 0,
) -> bool:
    """
    从source中移除包含任一关键词的行/bullet point。
    keywords: 触发移除的关键词列表（OR逻辑）。
    context_lines_to_skip: 前N个匹配跳过（保留上下文引用）。
    """
    source_lines = source_path.read_text(encoding="utf-8").splitlines()
    target_text = target_path.read_text(encoding="utf-8")
    
    # 确认target中确实包含这些关键词
    missing = [kw for kw in keywords if kw not in target_text]
    if missing:
        print(f"  ⚠ {target_path.name} 中未找到关键词: {missing}，跳过")
        return False
    
    removed = []
    kept = []
    skip_count = 0
    
    for i, line in enumerate(source_lines):
        matched_kws = [kw for kw in keywords if kw in line]
        if matched_kws:
            if skip_count < context_lines_to_skip:
                skip_count += 1
                print(f"  [保留-上下文] L{i+1}: {line.strip()[:70]}")
            else:
                removed.append((i + 1, line.strip(), matched_kws))
                print(f"  [移除] L{i+1}: {line.strip()[:80]}  (命中: {matched_kws})")
        else:
            kept.append(line)
    
    if not removed:
        print(f"  未发现需要移除的重复内容")
        return False
    
    source_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"  共移除 {len(removed)} 行")
    return True


def main():
    print(f"文档目录: {DOCS_DIR}\n")
    
    # === 去重规则 ===
    
    # 规则1: 移除cat_feeding.md中关于化毛膏/猫草的bullet point
    # pet_snacks.md有更完整的描述，cat_feeding.md保留布偶猫品种描述中的引用即可
    print("=" * 50)
    print("规则1: cat_feeding.md → pet_snacks.md (化毛膏/猫草)")
    print("=" * 50)
    dedup_by_keywords(
        DOCS_DIR / "cat_feeding.md",
        DOCS_DIR / "pet_snacks.md",
        keywords=["化毛膏", "猫草"],
        context_lines_to_skip=1,  # 保留布偶猫section中的首次引用
    )
    
    # 规则2: 移除cat_feeding.md中关于猫条引导进食的描述
    # pet_snacks.md有完整的猫条介绍
    print("\n" + "=" * 50)
    print("规则2: cat_feeding.md → pet_snacks.md (猫条)")
    print("=" * 50)
    dedup_by_keywords(
        DOCS_DIR / "cat_feeding.md",
        DOCS_DIR / "pet_snacks.md",
        keywords=["猫条"],
        context_lines_to_skip=0,
    )
    
    # 规则3: 移除dog_feeding.md中关于零食冻干的描述（如果有）
    print("\n" + "=" * 50)
    print("规则3: dog_feeding.md → pet_snacks.md (零食/冻干)")
    print("=" * 50)
    dedup_by_keywords(
        DOCS_DIR / "dog_feeding.md",
        DOCS_DIR / "pet_snacks.md",
        keywords=["零食", "冻干"],
        context_lines_to_skip=0,
    )
    
    print("\n" + "=" * 50)
    print("去重完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
