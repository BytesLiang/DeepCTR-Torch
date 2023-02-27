## DeepFm 

#### <1024 batch_size>

    LogLoss: 0.4767 AUC: 0.7691

#### <64 batch_size>

    LogLoss: 0.4749 AUC: 0.7719

## DCN

#### <1024 batch_size>
<embedding_dim=10>

    LogLoss: 0.4799 AUC: 0.7647
<embedding_dim=4>

    LogLoss: 0.4795 AUC: 0.7657

#### <64 batch_size>

    LogLoss: 0.4777 AUC: 0.7680

dense_embedding=10 sparse_embedding=4
0.4844 - val_auc:  0.7585
0.4847 - val_auc:  0.7585
0.4901 - val_auc:  0.7510
0.4845 - val_auc:  0.7588
0.4846 - val_auc:  0.7586
DeepFm
133s - loss:  0.4915 - binary_crossentropy:  0.4915 - auc:  0.7471 - val_binary_crossentropy:  0.4779 - val_auc:  0.7676
tau = e-1
318s - loss:  0.4922 - binary_crossentropy:  0.4922 - auc:  0.7474 - val_binary_crossentropy:  0.4759 - val_auc:  0.7699
tau = e-2
164s - loss:  0.4862 - binary_crossentropy:  0.4862 - auc:  0.7582 - val_binary_crossentropy:  0.4705 - val_auc:  0.7776
tau = e-3
160s - loss:  0.4887 - binary_crossentropy:  0.4887 - auc:  0.7545 - val_binary_crossentropy:  0.4690 - val_auc:  0.7797
tau = e-4
240s - loss:  0.4992 - binary_crossentropy:  0.4992 - auc:  0.7520 - val_binary_crossentropy:  0.4696 - val_auc:  0.7789
alpha = 0.1
225s - loss:  0.4850 - binary_crossentropy:  0.4850 - auc:  0.7598 - val_binary_crossentropy:  0.4678 - val_auc:  0.7813


alpha = 0.01
153s - loss:  0.4854 - binary_crossentropy:  0.4854 - auc:  0.7580 - val_binary_crossentropy:  0.4674 - val_auc:  0.7812
312s - loss:  0.4854 - binary_crossentropy:  0.4854 - auc:  0.7595 - val_binary_crossentropy:  0.4672 - val_auc:  0.7821
238s - loss:  0.4850 - binary_crossentropy:  0.4850 - auc:  0.7598 - val_binary_crossentropy:  0.4678 - val_auc:  0.7813
alpha = 0.001
149s - loss:  0.4867 - binary_crossentropy:  0.4866 - auc:  0.7566 - val_binary_crossentropy:  0.4679 - val_auc:  0.7807
30
204s - loss:  0.4850 - binary_crossentropy:  0.4850 - auc:  0.7597 - val_binary_crossentropy:  0.4674 - val_auc:  0.7817

kmean:
248s - loss:  0.4878 - binary_crossentropy:  0.4878 - auc:  0.7532 - val_binary_crossentropy:  0.4751 - val_auc:  0.7707
qu:
139s - loss:  0.4781 - binary_crossentropy:  0.4781 - auc:  0.7667 - val_binary_crossentropy:  0.4653 - val_auc:  0.7836
141s - loss:  0.4774 - binary_crossentropy:  0.4774 - auc:  0.7670 - val_binary_crossentropy:  0.4647 - val_auc:  0.7839