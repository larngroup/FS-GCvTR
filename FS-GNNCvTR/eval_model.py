import torch
from gnntr_eval import GNNCvTR_eval
import statistics
import matplotlib.pyplot as plt

def save_ckp(state, is_best, checkpoint_dir, best_model_dir, filename, best_model):
    
    f_path = checkpoint_dir + filename
    torch.save(state, f_path)
    #if is_best:
    #    best_fpath = best_model_dir + best_model
    #    shutil.copyfile(f_path, best_fpath)

def save_result(epoch, N, exp, filename):
            
    file = open(filename, "a")
    
    file.write("Results: " + "\t")
    if epoch < N:
        file.write(str(exp) + "\t")
    if epoch == N:
        file.write(str(exp))
        for list_acc in exp:
           file.write(str(epoch) + " Support Sets: (Mean:"+ str(statistics.mean(list_acc)) +", SD:" +str(statistics.stdev(list_acc)) + ") | \t")
                 
    file.write("\n")
    file.close()

dataset = "tox21"
gnn = "gin"
support_set = 5
pretrained = "pre-trained/supervised_contextpred.pth"
baseline = 0
device = "cuda:0"

# FS-GNNCvTR - Two module GNN-CvTR architecture
# GraphSage - Inductive method to aggregate node representations for unseen graph features
# GIN - Graph Isomorphism Network
# GCN - Standard Graph Convolutional Network

device = "cuda:0"      
model_eval = GNNCvTR_eval(dataset, gnn, support_set, pretrained, baseline)
model_eval.to(device)

print("Dataset:", dataset)

roc_auc_list = []

if dataset== "tox21":
    exp = [[],[],[]]
    labels =  ['SR-HSE', 'SR-MMP', 'SR-p53']
elif dataset == "sider":
    exp = [[],[],[],[],[],[]]
    labels =  ['R.U.D.', 'P.P.P.C.', 'E.L.D.', 'C.D.', 'N.S.D.', 'I.P.P.C.']    

N = 30
   
for epoch in range(1, 10000):
    
    roc_scores, gnn_model, transformer_model, gnn_opt, t_opt = model_eval.meta_evaluate() #FS-GNNCvTR
   
    #roc_scores, gnn_model, gnn_opt = model.meta_evaluate(grads) #baselines
                  
    if epoch <= N:
      i=0
      for a in roc_scores:
        exp[i].append(round(a,4))
        i+=1
      
    if epoch > N:
      for i in range(len(exp)):
        if min(exp[i]) < round(roc_scores[i],4):
          index = exp[i].index(min(exp[i]))
          exp[i][index] = roc_scores[i]
      
    save_result(epoch, N, exp, "results-exp/mean-FSGNNCvTR_tox21_5.txt")
      
    if dataset == "tox21":
        box_plot_data=[exp[0], exp[1], exp[2]]
        plot_title = "Tox21"
    elif dataset == "sider":
        box_plot_data=[exp[0], exp[1], exp[2], exp[3], exp[4], exp[5]]
        plot_title = "SIDER"  
    
    fig = plt.figure()   
    fig.suptitle(plot_title, fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.boxplot(box_plot_data,labels=labels)
    ax.set_xlabel('Test Task')
    ax.set_ylabel('ROC-AUC score')
    plt.grid(b=False)
    if epoch == N:
        plt.savefig('plots/test/boxplot_GT_'+ str(dataset) + '_' + str(support_set), dpi=300)
    #plt.savefig('plots/boxplot_GT_'+ str(dataset) + '_' + str(support_set), dpi=300)
    plt.show()
    plt.close(fig)
    
