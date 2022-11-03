import numpy as np
# from utils.tools import plot_embedding, get_configs_of
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import os


def plot_embedding(out_dir, embedding, embedding_accent_id,colors,markers,labels,filename='embedding.png'):
#    colors = 'r','b','g','y'
#    labels = 'Female','Male'

    data_x = embedding
    data_y = embedding_accent_id
#    data_y = np.array([gender_dict[spk_id] == 'M' for spk_id in embedding_speaker_id], dtype=np.int)
    tsne_model = TSNE(n_components=2, random_state=0, init='random')
    tsne_all_data = tsne_model.fit_transform(data_x)
    tsne_all_y_data = data_y

    plt.figure(figsize=(10,10))
    if markers is not None:
        for i, (c, label, mark) in enumerate(zip(colors, labels, markers)):
            plt.scatter(tsne_all_data[tsne_all_y_data==i,0], tsne_all_data[tsne_all_y_data==i,1], c=c, marker=mark, label=label, alpha=0.5)
    else:
        for i, (c, label) in enumerate(zip(colors, labels)):
            plt.scatter(tsne_all_data[tsne_all_y_data==i,0], tsne_all_data[tsne_all_y_data==i,1], c=c, label=label, alpha=0.5)

    plt.grid(True)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename))


# colors = 'r','b','g','y','k','c'
# colors2 = 'r','b','g','y','k','c','r','b','g','y','k','c','r','b','g','y','k','c','r','b','g','y','k','c'

# markers2 = 'g','r','r','c','g','b','b','m','b','c','m','b','y','g','m','c','g','r','y','m','c','r','y','y'
# colors2 = ['g','r','r','c','g','b','b','m','b','c','m','b','y','g','m','c','g','r','y','m','c','r','y','y']


# markers2 = 'g1','r1','r3','c1','g3','b3','b1','m1','b2','c2','m3','b4','y1','g2','m4','c3','g4','r4','y3','m2','c4','r2','y4','y2'

# markers2 = '1','1','3','1','3','3','1','1','2','2','3','4','1','2','4','3','4','4','3','2','4','2','4','2'

# markers2 = ['x','x','v','x','v','v','x','x','+','+','v','o','x','+','o','v','o','o','v','+','o','+','o','+']


# preprocess_config, model_config, train_config = get_configs_of("L2Arctic")
# labels = preprocess_config["accents"]
labels = ['Arabic', 'Chinese', 'Hindi', 'Korean', 'Spanish', 'Vietnamese']
#PLEASE NOTE MY ACCENTS ARE IN ALPHABETICAL ORDER! IT'S LIKE THAT IN ALL MY CONFIG FILES (EXCEPT FOR THE COPIED GMVAE)
#-> ARABIC, CHINESE, HINDI, KOREAN, SPANISH, VIETNAMESE!

# spk_lab = {"ABA", "SKA", "YBAA", "ZHAA", "BWC", "LXC", "NCC", "TXHC", "ASI", "RRBI", "SVBI", "TNI", "HJK", "HKK", "YDCK", "YKWK", "EBVS", "ERMS", "MBMPS", "NJS", "HQTV", "PNV", "THV", "TLV"}
spk_lab = ["RRBI", "ABA", "SKA", "EBVS", "TNI", "NCC", "BWC", "HQTV", "TXHC", "ERMS", "PNV", "LXC", "HKK", "ASI", "THV", "MBMPS", "SVBI", "ZHAA", "HJK", "TLV", "NJS", "YBAA", "YDCK", "YKWK"]


out_dir='output/plots'
acc_embed=np.load('output/arrays/acc_mu.npy')
spk_embed=np.load('output/arrays/spk_mu.npy')
embedding_acc_id=np.load('output/arrays/acc_id.npy')
embedding_spk_id=np.load('output/arrays/spk_id.npy')


m1='x' #first male
m2='+' #second male
m3='v' #first female
m4='o' #second female

markers2 = [m1,m1,m3,m1,m3,m3,m1,m1,m2,m2,m3,m4,m1,m2,m4,m3,m4,m4,m3,m2,m4,m2,m4,m2] #marker for each speaker based on gender and order

ara='r'
chi='b'
hin='g'
kor='y'
spa='c'
vie='m'

colors = ara,chi,hin,kor,spa,vie #just accent colors

colors2 = [hin,ara,ara,spa,hin,chi,chi,vie,chi,spa,vie,chi,kor,hin,vie,spa,hin,ara,kor,vie,spa,ara,kor,kor] #colors for each speaker based on accent


#PICK ONLY CERTAIN PEOPLEEEEEEEEE
indexlist=[0,1,2,3,4,5,6,7,10,12,15,18]
noindexlist=[]
nospklist=[]
for k, spk in enumerate(spk_lab):
    if k in indexlist:
        continue
    else:
        embedding_acc_id=embedding_acc_id[embedding_spk_id!=k]
        acc_embed=acc_embed[embedding_spk_id!=k,:]
        spk_embed=spk_embed[embedding_spk_id!=k,:]
        embedding_spk_id=embedding_spk_id[embedding_spk_id!=k]
        nospklist.append(spk)
        noindexlist.append(k)


spk_lab = [spk for spk in spk_lab if spk not in nospklist]



c2=colors2.copy()
m2=markers2.copy()

colors2=[]
markers2=[]

for i,(c,m) in enumerate(zip(c2,m2)):
    if i in indexlist:
        colors2.append(c)
        markers2.append(m)

colors2=tuple(colors2)
markers2=tuple(markers2)

#map old IDs to new IDs through this unique list mapping, otherwise plotting had issues!
spk_unique=np.unique(embedding_spk_id)

for i,spk_id in enumerate(spk_unique):
    embedding_spk_id[embedding_spk_id==spk_id]=i









plot_embedding(out_dir, acc_embed, embedding_acc_id,colors,None,labels,filename='embedding_acc.png')
plot_embedding(out_dir, spk_embed, embedding_spk_id,colors2,markers2,spk_lab,filename='embedding_spk.png')

plot_embedding(out_dir, acc_embed, embedding_spk_id,colors2,markers2,spk_lab,filename='embedding_acc_spklab.png')
# plot_embedding(out_dir, spk_embed, embedding_spk_id,colors2,markers2,spk_lab,filename='embedding_spk.png')

plot_embedding(out_dir, np.concatenate((acc_embed,spk_embed),1), embedding_acc_id,colors,None,labels,filename='embedding_combined_acc.png')
plot_embedding(out_dir, np.concatenate((acc_embed,spk_embed),1), embedding_spk_id,colors2,markers2,spk_lab,filename='embedding_combined_spk.png')