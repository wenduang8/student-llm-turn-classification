# extraction/：语料是怎样构建的

语料在2026年9月冻结。这里的脚本记录构建方法，不要重跑：原始提交只在本机，第2步调用Claude API，重跑还会覆盖现有语料。

## 实际执行顺序

1. `authentic_extractor.py`：从原始提交（多种格式）提取对话并规范化，写 extraction_manifest、conversations_master 和按说话人标记切分的 transcripts_segmented.csv。PDF文本经 `pdf_extractor.extract_pdf_text`。
2. `segment_needs_review.py --all --apply`：没有说话人标记的对话由Claude API切分，结果缓存。
3. `recover_manual_rules.py`：少数学生的固定格式、MHTML快照和手工标注的切分。
4. `review_report.py`：把剩下的待复核文件分类并锁定。
5. `combine_dedup.py`：合并三种来源并去重，得到 transcripts_all.csv。这是修复前的基线（6,440条），现保留为 transcripts_all_v1_pre_repair_20260916.csv。
6. `repair_openings.py --write`：补回提取器漏掉的131段对话开头（D-53），写 transcripts_all_v2.csv；核对后它成为现在的 transcripts_all.csv（6,595条，1,204段对话）。
7. `apply_d60_exclusion.py`：排除两段近似副本对话（D-60），写 transcripts_analysis.csv（6,567条，1,202段对话，361名学生；编码和训练用的分析文件）和 exclusions_d60.csv。transcripts_all.csv 不改。

## 只读核对（随时可运行）

- `python extraction/validate_corpus_v2.py`：修复后的 transcripts_all.csv 与修复前备份逐项比对。
- `python extraction/apply_d60_exclusion.py --check`：重算排除后的数字并与D-60登记的比对，不写文件。
- `scan_share_links.py`：语料中聊天链接的审计，是D-60认定近似副本的依据。它写到具体对话，不随公开版发布。

## 已知局限

不影响485条标注和论文数字，因为标签和模型读的是同一份文本。

- 第5步去重后把 Segment Order 重新编号，去重删掉的行不留空缺；只有第7步之前的无回复行排除会留下空缺。涉及12段对话（分析文件里11段），485条里没有。
- 第2步图片组的对话编号（`_IMG####`）由 Python 的 `hash()` 生成，每次运行不同；语料里的13个来自构建语料的那次运行，不能再生成。
- 第2步锚点找不到原文时用近似位置：42个回退边界里41个提前1至882个字符（中位数4），涉及12段对话；485条里1条受影响，最多3个字符。

## 已归档（D-203）

`run_all.py`（旧的一条命令流水线，不含第6、7步）、`verify_authentic.py`（检查修复前基线的不变量）、`audit_coverage.py`、`verify_extraction.py` 在 `docs/archive/code/extraction/`，路径对照见 `data/maintenance/20261028-code-review/path-map.csv`。
