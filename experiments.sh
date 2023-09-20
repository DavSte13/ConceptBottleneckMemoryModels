

# ======================= Source code, datasets and preprocessing =======================
# You should have `CUB_200_2011`, `CUB_processed`, `places365`, `pretrained`, `src` all available on the path during experiment runs.
# Each python script outputs to a folder, change `-log_dir` or `-out_dir` if you would like different output folders.

# Experiments

## ======================= Main experiments =======================
### Concept Model
python3 experiments.py cub Bottleneck -log_dir results/ConceptModel_fold_0/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 1000 -bottleneck
python3 CUB/generate_new_data.py ExtractConcepts -model_path results/ConceptModel_fold_0/outputs/best_model.pth -out_dir ConceptModel1__PredConcepts -data_dir = data_CUB/CUB_processed/class_filtered_10

### Independent Model
python3 experiments.py cub Independent -log_dir results/IndependentModel/outputs/ -e 500 -optimizer sgd -no_img -b 64 -weight_decay 0.00005 -lr 0.001 -scheduler_step 1000
python3 inference.py -model_dir results/ConceptModel_fold_0/outputs/best_model.pth -model_dir2 results/IndependentModel/outputs/best_model.pth -eval_data test -bottleneck -use_sigmoid -log_dir results/IndependentModel/outputs

### Sequential Model
python3 experiments.py cub Sequential -log_dir results/SequentialModel_fold_0/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -data_dir ConceptModel1__PredConcepts -no_img -b 64 -weight_decay 0.00004 -lr 0.001 -scheduler_step 1000
python3 inference.py -model_dir results/ConceptModel_fold_0/outputs/best_model.pth -model_dir2 results/SequentialModel_fold_0/outputs/best_model.pth -eval_data test -bottleneck -log_dir results/SequentialModel/outputs

### Standard Model
python3 experiments.py cub Joint -ckpt -log_dir results/Joint0Model_Seed1/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -attr_loss_weight 0 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 20 -end2end
python3 inference.py -model_dir results/Joint0Model_Seed1/outputs/best_model.pth -eval_data test -log_dir results/Joint0Model/outputs

### Joint Model
#### Concept loss weight = 0.001
python3 experiments.py cub Joint -ckpt -log_dir results/Joint0.001Model_Seed1/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -attr_loss_weight 0.001 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 20 -end2end
python3 inference.py -model_dir results/Joint0.001Model_Seed1/outputs/best_model.pth -eval_data test -log_dir results/Joint0.001Model/outputs
#### Concept loss weight = 0.01
python3 experiments.py cub Joint -ckpt -log_dir results/Joint0.01Model_fold_0/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.001 -scheduler_step 1000 -end2end
python3 inference.py -model_dir results/Joint0.01Model_fold_0/outputs/best_model.pth -eval_data test -log_dir results/Joint0.01Model/outputs
#### Concept loss weight = 0.01, with Sigmoid
python3 experiments.py cub Joint -ckpt -log_dir results/Joint0.01SigmoidModel_fold_0/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.001 -scheduler_step 1000 -end2end -use_sigmoid
#### Concept loss weight = 0.1
python3 experiments.py cub Joint -ckpt -log_dir results/Joint0.1Model_fold_0/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -attr_loss_weight 0.1 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 1000 -end2end
python3 inference.py -model_dir results/Joint0.1Model_fold_0/outputs/best_model.pth -eval_data test -log_dir results/Joint0.1Model/outputs
#### Concept loss weight = 1
python3 experiments.py cub Joint -ckpt -log_dir Joint1Model_fold_0/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -attr_loss_weight 1 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 1000 -end2end
python3 inference.py -model_dir Joint1Model_fold_0/outputs/best_model.pth -eval_data test -log_dir Joint1Model/outputs

## ======================= Test-time intervention experiments =======================
### Independent Model
python3 CUB/tti.py -model_dir results/ConceptModel_fold_0/outputs/best_model.pth -model_dir2 results/IndependentModel/outputs/best_model.pth -bottleneck -mode random -use_invisible -use_sigmoid -log_dir results/TTI__IndependentModel

### Sequential Model
python3 CUB/tti.py -model_dir results/ConceptModel_fold_0/outputs/best_model.pth -model_dir2 results/SequentialModel_fold_0/outputs/best_model.pth -bottleneck -mode random -use_invisible -log_dir results/TTI__SequentialModel

### Joint Model
python3 CUB/tti.py -model_dir results/Joint0.01SigmoidModel_fold_0/outputs/best_model.pth -use_sigmoid -mode random -use_invisible -log_dir results/TTI__results/Joint0.01SigmoidModel
python3 CUB/tti.py -model_dir results/Joint0.01Model_fold_0/outputs/best_model.pth -mode random -use_invisible -log_dir results/TTI__results/Joint0.01Model

## ======================= Robustness experiments =======================
### Generate Adversarial Data
python3 CUB/gen_cub_synthetic.py -cub_dir CUB_200_2011 -places_dir places365 -out_dir data_CUB/AdversarialData
### Generate pickle files from the adversarial data
python3 CUB/generate_new_data.py ChangeAdversarialDataDir -adv_data_dir data_CUB/AdversarialData/CUB_fixed -train_splits train val -out_dir data_CUB/CUB_processed/adversarial_fixed -data_dir data_CUB/CUB_processed/class_filtered_10
python3 CUB/generate_new_data.py ChangeAdversarialDataDir -adv_data_dir data_CUB/AdversarialData/CUB_random -train_splits train val -out_dir data_CUB/CUB_processed/adversarial_random -data_dir data_CUB/CUB_processed/class_filtered_10
python3 CUB/generate_new_data.py ChangeAdversarialDataDir -adv_data_dir data_CUB/AdversarialData/CUB_black -train_splits train val -out_dir data_CUB/CUB_processed/adversarial_black -data_dir data_CUB/CUB_processed/class_filtered_10

### Concept Model
python3 experiments.py cub Bottleneck -ckpt -log_dir ConceptAdversarialModel_fold_0/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 1000 -bottleneck -image_dir data_CUB/AdversarialData/CUB_fixed/train -data_dir data_CUB/CUB_processed/adversarial_fixed
python3 CUB/generate_new_data.py ExtractConcepts -model_path ConceptAdversarialModel_fold_0/outputs/best_model.pth -data_dir data_CUB/CUB_processed/adversarial_fixed -out_dir ConceptAdversarialModel1__PredConcepts -data_dir data_CUB/CUB_processed/adversarial_fixed

### Independent Model
python3 experiments.py cub Independent -log_dir IndependentAdversarialModel/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -no_img -b 64 -weight_decay 0.00005 -lr 0.001 -scheduler_step 1000 -image_dir data_CUB/AdversarialData/CUB_fixed/train -use_sigmoid -data_dir data_CUB/CUB_processed/adversarial_fixed
python3 inference.py -model_dir ConceptAdversarialModel_fold_0/outputs/best_model.pth -model_dir2 IndependentAdversarialModel/outputs/best_model.pth -eval_data test -bottleneck -image_dir data_CUB/AdversarialData/CUB_fixed/test/ -use_sigmoid -log_dir IndependentAdversarialModel/outputs -data_dir data_CUB/CUB_processed/adversarial_fixed

### Sequential Model
python3 experiments.py cub Sequential -log_dir SequentialAdversarialModel_fold_0/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -data_dir ConceptAdversarialModel1__PredConcepts -no_img -b 64 -weight_decay 0.00004 -lr 0.001 -scheduler_step 1000 -image_dir data_CUB/AdversarialData/CUB_fixed/train -data_dir data_CUB/CUB_processed/adversarial_fixed
python3 inference.py -model_dir ConceptAdversarialModel_fold_0/outputs/best_model.pth -model_dir2 SequentialAdversarialModel_fold_0/outputs/best_model.pth -eval_data test -bottleneck -image_dir data_CUB/AdversarialData/CUB_fixed/test/ -log_dir SequentialAdversarialModel/outputs -data_dir data_CUB/CUB_processed/adversarial_fixed

### Standard Model
python3 experiments.py cub Joint -ckpt -log_dir results/Joint0AdversarialModel_fold_0/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -attr_loss_weight 0 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.001 -scheduler_step 1000 -end2end -image_dir data_CUB/AdversarialData/CUB_fixed/train -data_dir data_CUB/CUB_processed/adversarial_fixed
python3 inference.py -model_dir results/Joint0AdversarialModel_fold_0/outputs/best_model.pth -eval_data test -image_dir data_CUB/AdversarialData/CUB_fixed/test/ -log_dir results/Joint0AdversarialModel/outputs -data_dir data_CUB/CUB_processed/adversarial_fixed

### Joint Model
python3 experiments.py cub Joint -ckpt -log_dir results/Joint0.01AdversarialModel_fold_0/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.001 -scheduler_step 1000 -end2end -image_dir data_CUB/AdversarialData/CUB_fixed/train -data_dir data_CUB/CUB_processed/adversarial_fixed
python3 inference.py -model_dir results/Joint0.01AdversarialModel_fold_0/outputs/best_model.pth -eval_data test -image_dir data_CUB/AdversarialData/CUB_fixed/test/ -log_dir results/Joint0.01AdversarialModel/outputs -data_dir data_CUB/CUB_processed/adversarial_fixed

