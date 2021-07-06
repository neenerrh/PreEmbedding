# Implement BPR.
# Steffen Rendle, et al. BPR: Bayesian personalized ranking from implicit feedback.
# Proceedings of the twenty-fifth conference on uncertainty in artificial intelligence. AUAI, 2009. 
# @author Runlong Yu, Mingyue Cheng, Weibo Gao

import random
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
#import scores
import math
from scipy.special import expit

class BPR:
    user_count =  32242
    item_count =  30066
    #user_count = 11955
    #item_count = 103
    #user_count = 25667
    #item_count = 145
    latent_factors = 10
    lr = 0.025
    reg = 0.01
    train_count =15
    train_data_path = r'training3.txt'
    test_data_path = r'test3.txt'
    user_embedding_file = r'u_vectors.txt'
    item_embedding_file = r'v_vectors.txt'
    size_u_i = user_count * item_count
    # latent_factors of U & V
    U = np.random.rand(user_count, latent_factors) * 0.01  
    V = np.random.rand(item_count, latent_factors) * 0.01
    biasV = np.random.rand(item_count) * 0.01
    
    #test_data = np.zeros((user_count, item_count))
    #test = np.zeros(size_u_i)
    #test= {}
    #predict={}
    #predict_ = np.zeros(size_u_i)
        
        

    def load_data(self, path):
        user_ratings = defaultdict(set)
        items=[]
        with open(path, 'r') as f:
            for line in f.readlines():
                u, i ,rate= line.strip().split("\t")
                u = 'u'+ u
                i = 'i'+ i
                items.append(i)
                user_ratings[u].add(i)
        res = [] 
        [res.append(x) for x in items if x not in res]         
        return user_ratings,res

    def load_test_data(self, path):
        items=[]
        file = open(path, 'r')
        for line in file:
            line = line.strip().split("\t")
            user = line[0]
            item = line[1]
            if item not in set(items):
                items.append(item)
            self.test_data['user']['item'] = 1
        res = [] 
        [res.append(x) for x in items if x not in res]  
        return res
    
    def read_data(self,filename):
        if filename is None:
            filename = os.path.join(self.model_path,"ratings_test.dat")
        users,items,rates = set(), set(), {}
        with open(filename, "r", encoding="UTF-8") as fin:
            line = fin.readline()
            while line:
                user, item, time= line.split('\t')
                if rates.get(user) is None:
                    rates[user] = {}
                rates[user][item] = 1
                users.add(user)
                items.add(item)
                line = fin.readline()
        return users, items, rates


    def train(self, user_ratings_train,users,train_users,items,train_items):
       
        #print(self.U)
        #print(self.U['U_1201914'])
        
        users=users
        train_users=list(train_users)
        items=items
        
        train_items=train_items
        #print(dict(user_ratings_train))
        #print(train_users)
        loss=0
        for user in range(self.user_count):
            # sample a user
            #u = random.randint(1, self.user_count)
            
            u = random.sample(users,1)
           
            u=u[0]
            
            if u not in train_users:              
                continue
          
            
            i = random.sample(user_ratings_train[u], 1)
            i=i[0]
            
            # sample a negative item from the unobserved items
            j = random.sample(items,1)
            j=j[0]
            while j in user_ratings_train[u]:
                j = random.sample(items,1)
                j=j[0]
            user_u = self.U[str(u)]
            item_i = self.V[str(i)]
            item_j = self.V[str(j)]    
            
            self.U[str(u)].setflags(write=1)
            self.V[str(i)].setflags(write=1)
            self.V[str(j)].setflags(write=1) 
            
                    
            
           
            #Method 1
             
            ##dot product
            #self.r_ui = np.dot(self.U.get(str(u)), (self.V.get(str(i))).T) + self.biasV.get(str(i))
            #self.r_uj = np.dot(self.U.get(str(u)), (self.V.get(str(j))).T) + self.biasV.get(str(j))
            #self.r_uij = self.r_ui - self.r_uj
            #self.loss_func = -1.0 / (1 + np.exp(self.r_uij)) 
            
            #sigmod=1.0 / (1 + np.exp(-self.r_uij)) 
           
                     
            ##update U and V           
            
            #self.U[str(u)] += -self.lr * (self.loss_func * (item_i - item_j) + self.reg * user_u)
            #self.V[str(i)] += -self.lr * (self.loss_func * user_u + self.reg * item_i)
            #self.V[str(j)] += -self.lr * (self.loss_func * (-user_u) + self.reg * item_j)
            #update biasV
            #self.biasV[str(i)] += -self.lr * (self.loss_func + self.reg * self.biasV[str(i)])
            #self.biasV[str(j)] += -self.lr * (-self.loss_func + self.reg * self.biasV[str(j)])
            
           
            
            #Method2
             
            #dot product
            self.r_ui = np.dot(self.U.get(str(u)), (self.V.get(str(i))).T) 
            self.r_uj = np.dot(self.U.get(str(u)), (self.V.get(str(j))).T) 
            self.r_uij = self.r_ui - self.r_uj
            
            self.sigmoid =np.exp(-self.r_uij) / (1.0 + np.exp(-self.r_uij)) 
            
            # update using gradient descent            
            #self.sigmod = expit(self.r_uij)
            self.sigmod=1/(1.0+np.exp(-self.r_uij))
            
            #self.grad_u = self.loss_func * (item_i - item_j) + self.reg * user_u
            #self.grad_i = self.loss_func * user_u + self.reg * item_i
            #self.grad_j = self.loss_func * -user_u + self.reg * item_j
            
            #self.sigmoid= expit(self.r_uij)
                           
            self.grad_u = self.sigmoid * (self.V[str(j)] - self.V[str(i)]) -self.reg * self.U[str(u)]
            self.grad_i = self.sigmoid * (-self.U[str(u)]) -self.reg * self.V[str(i)]
            self.grad_j = self.sigmoid * (self.U[str(u)]) -self.reg * self.V[str(j)]
            
            
            self.U[str(u)] -= self.lr * self.grad_u 
            self.V[str(i)] -= self.lr * self.grad_i
            self.V[str(j)] -= self.lr * self.grad_j
            
            self.loss= math.log(self.sigmod) 
            
            loss+= self.loss
        #print(loss)      
        return(loss)   
        #print(self.U)
        #print(self.V)
        
    def top_N(self,test_u, test_v, test_rate, vectors_u, vectors_v, top_n):
        predict = {}
        #
       
        #print(vectors_u)
        for u in test_u:
            predict[u] = {}
            for v in test_v:
                if vectors_u.get(u) is None:
                    pre = 0
                else:
                    
                    #U = np.array(vectors_u[u].keys())
                    U=vectors_u.get(u)
                         
                    #print(U.dtype)
                    if vectors_v.get(v) is None:
                        pre = 0
                    else:
                        #V = np.array(vectors_v[v].keys())
                        #pre = U.dot(V.T)
                        V=vectors_v.get(v)
                        V=np.transpose(V)
                        #print(V)
                        #print(V.dtype)
                        pre= np.dot(U,V)
                predict[u][v] = float(pre)
            
   

        precision_list = []
        recall_list = []
        ap_list = []
        ndcg_list = []
        rr_list = []

        for u in test_u:
        

            from functools import cmp_to_key
            def cmp(x, y):                   # emulate cmp from Python 2
                if (x< y):
                    return -1
                elif (x == y):
                    return 0
                elif (x > y):
                    return 1
            tmp_r = sorted(predict[u].items(), key=cmp_to_key(lambda x, y: cmp(x[1], y[1])), reverse=True)[0:min(len(predict[u]),top_n)]
            tmp_t = sorted(test_rate[u].items(), key=cmp_to_key(lambda x, y: cmp(x[1], y[1])), reverse=True)[0:min(len(test_rate[u]),top_n)]
        
        
            tmp_r_list = []
            tmp_t_list = []
            for (item, rate) in tmp_r:
                tmp_r_list.append(item)

            for (item, rate) in tmp_t:
                tmp_t_list.append(item)
            pre, rec = self.precision_and_recall(tmp_r_list,tmp_t_list)
            ap = self.AP(tmp_r_list,tmp_t_list)
            rr = self.RR(tmp_r_list,tmp_t_list)
            ndcg = self.nDCG(tmp_r_list,tmp_t_list)
            precision_list.append(pre)
            recall_list.append(rec)
            ap_list.append(ap)
            rr_list.append(rr)
            ndcg_list.append(ndcg)
        precison = sum(precision_list) / len(precision_list)
        recall = sum(recall_list) / len(recall_list)
    #print(precison, recall)
        f1 = 2 * precison * recall / (precison + recall)
        map = sum(ap_list) / len(ap_list)
        mrr = sum(rr_list) / len(rr_list)
        mndcg = sum(ndcg_list) / len(ndcg_list)
        return f1,map,mrr,mndcg,recall
    
    def nDCG(self,ranked_list, ground_truth):
        dcg = 0
        idcg =self. IDCG(len(ground_truth))
        for i in range(len(ranked_list)):
            id = ranked_list[i]
            if id not in ground_truth:
                continue
            rank = i+1
            dcg += 1/ math.log(rank+1, 2)
        return dcg / idcg

    def IDCG(self,n):
        idcg = 0
        for i in range(n):
            idcg += 1 / math.log(i+2, 2)
        return idcg

    def AP(self,ranked_list, ground_truth):
        hits, sum_precs = 0, 0.0
        for i in range(len(ranked_list)):
            id = ranked_list[i]
            if id in ground_truth:
                hits += 1
                sum_precs += hits / (i+1.0)
        if hits > 0:
            return sum_precs / len(ground_truth)
        else:
            return 0.0

    def RR(self,ranked_list, ground_list):

        for i in range(len(ranked_list)):
            id = ranked_list[i]
            if id in ground_list:
                return 1 / (i + 1.0)
        return 0

    def precision_and_recall(self,ranked_list,ground_list):
        hits = 0
        for i in range(len(ranked_list)):
            id = ranked_list[i]
            if id in ground_list:
                hits += 1
        pre = hits/(1.0 * len(ranked_list))
        rec = hits/(1.0 * len(ground_list))
        return pre, rec 

    def predict(self, user, item):
        #predict = np.mat(user) * np.mat(item.T)
        predict = user * (item.T)
        return predict
    
    def save_user_embeddings(self, U,filename):
        fout = open(filename, 'w')
        node_num = len(U.keys())
        fout.write("{} {}\n".format(node_num, self.latent_factors))
        for node, vec in U.items():
            fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
        fout.close()
        
        
    def save_item_embeddings(self, V, filename):
        fout = open(filename, 'w')
        node_num = len(V.keys())
        fout.write("{} {}\n".format(node_num, self.latent_factors))
        for node, vec in V.items():
            fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
        fout.close()

    def main(self):
        user_ratings_train,train_items= self.load_data(self.train_data_path)
        #print(user_ratings_train)
        train_users=user_ratings_train.keys() 
        test_users,test_items,test_data =self.read_data(self.test_data_path)
        all_users= list(train_users) + list(test_users)
        res1 = [] 
        [res1.append(x) for x in all_users if x not in res1]
        #all users list
        users=res1        
        items=train_items + list(test_items)
        res2 = [] 
        [res2.append(x) for x in items if x not in res2]
        #all item list
        items=res2       
        print(len(users))
        print(len(items))
        self.U= dict(zip(users,self.U))
        self.V=dict(zip(items,self.V))
        self.biasV=dict(zip(items,self.biasV))
        print("training")
        
        last_loss=0
        #last_loss=0
        for i in range(self.train_count):
            loss=self.train(user_ratings_train,users,train_users,items,train_items)
            last_loss+=loss
            #print(self.sigmoid)
            delta_loss=abs(last_loss-self.loss)
            
            print(loss)
            #print(last_loss)
            #print(delta_loss)
            #last_loss=self.loss
            #if delta_loss < 0.1:
                #break
        print("results")
        
        self.save_user_embeddings(self.U,self.user_embedding_file)
        self.save_item_embeddings(self.V,self.item_embedding_file)
        
        f1, map, mrr, mndcg,recall = self.top_N(test_users, test_items, test_data, self.U, self.V, 10)
        print("done")
       
        print('recommendation metrics: F1 : %0.4f, MAP : %0.4f, MRR : %0.4f, NDCG : %0.4f, RECALL: %0.4f' % (round(f1,4), round(map,4), round(mrr,4), round(mndcg,4),round(recall,4)))
        #predict_matrix = self.predict(self.U, self.V)
        # prediction
        #self.predict_ = predict_matrix.getA().reshape(-1)
        #self.predict_ = pre_handel(user_ratings_train, self.predict_, self.item_count)
        #auc_score = roc_auc_score(self.test, self.predict_)
        #print('AUC:', auc_score)
        # Top-K evaluation
        #scores.topK_scores(self.test, self.predict_, 5, self.user_count, self.item_count)

def pre_handel(set, predict, item_count):
    # Ensure the recommendation cannot be positive items in the training set.
    for u in set.keys():
        for j in set[u]:
            predict[(u - 1) * item_count + j - 1] = 0
    return predict

if __name__ == '__main__':
    bpr = BPR()
    bpr.main()
