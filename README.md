# DML-project

Bebbe får hela jävla johannebergs berggrund att krackelera när han pullar upp till campus #true

Länkar:
Projek Instruc: https://canvas.chalmers.se/courses/40843/pages/project-instructions?module_item_id=706491

Data Sets:
PTB-XL: https://physionet.org/content/ptb-xl/1.0.3/

Workflow after dataloader and datasets are constructed

Step 1 CNN model class Conv1d with 12 inputs, 5 outputs; test shapes with a fake batch
Step 2 Training loop multi-label loss + macro AUC, with an optional lead-dropout switch
Step 3 Train model A dropout off; save the trained weights to disk
Step 4 Evaluation with masks test A on validation: all 12 leads vs. 3 leads
Step 5 Train model B the same code with dropout on; save the weights
Step 6 Compare on validation choose the dropout rate here, never on test
Step 7 Final test runs A and B on 12 leads, on I/II/V2, and on all 220 triplets
Step 8 Analysis and plots which leads matter, which diagnoses suffer most
