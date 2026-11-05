# coding/: human coding

These scripts produced the human coding used in the thesis: choosing the conversations to code, building the coding packets, receiving and checking the returned records, and comparing two coders. They run only on the private data under `data/`: the modules name fixed folders there, and the packet builder refuses to run unless the corpus and `codebook_schema.json` match the baseline the packets were built from. The tests run on synthetic records.

## Order

1. `select_materials_v2.py`: the allocation engine. It selects whole conversations, keeps the students of different purposes (evaluation, development, training) apart, and records why each excluded conversation was excluded. It imports the pool, ranking and first-batch functions from `select_materials.py`, the earlier engine it replaced, which is kept as that library and as the baseline its tests compare against.
2. `build_direct_packet.py`: builds a de-identified coding packet (a reader and a response form) with an identifier screen. `comparison/deidentify.py` reuses the screen.
3. `receive_return.py`: stores a returned packet unchanged and writes a receipt.
4. `direct_coding.py`: validates human coding records against the schema. The other scripts and `finetuning/labels.py` use it.
5. `compare_coders.py`: a workbook of the two coders' differences, for their discussion. It computes no agreement figure.
6. `coder_agreement.py`: per-code agreement C / max(A, B) between two coders before discussion (Table 4.1 of the thesis), and Cohen's kappa for the student interaction type.

The interface text of the packets and some printed output are in Chinese, the working language of the coding.

## Known limitation

The identifier screen does not catch some forms: a lower-case Windows user folder, an upper-case unikey or one followed by an underscore, and a phone number written with dashes. A recheck of the 485 labelled interactions found none of these forms.
