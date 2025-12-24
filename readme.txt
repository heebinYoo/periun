# (AAAI26 Poster) Hee Bin Yoo, Dong-Sig Han, Jaein Kim, and Byoung-Tak Zhang. “PeriUn: Enhancing Unlearning by Selectively Forgetting Peripheral Samples”.

1. go to setup folder and run expr_setup and model_data_setup 
2. run train_orig and train_retrain 
3. go to acc_evaluation and run baseline_performance_calc 
4. go to setup folder and run salun_setup 
5. Run hyperparameter_search to find hyperparameter (need to access wandb) 
6. Download the wandb result and run tow_calc in acc_evaluation 
7. Get hyperparameters 8. Run repeat_expr to get ToW results 
9. Run mia_eval to obtain MIA gap results
10. Run observation_one to get the first result in Preliminary Analysis of the Retrained Model
11. Run observation_two to get the second result in Preliminary Analysis of the Retrained Model
12. Run tsne_analysis to get the TSNE analysis in the Preliminary Analysis of the Retrained Model
13. Run jsd_analysis to get JSD results
14. go to quantative_analysis folder and run quantative_analysis to get quantative analysis in Experiment
