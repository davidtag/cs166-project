from lsh import *
from utils import *

class query:
    def __init__(self, data=None, queries_dataset=None, hash_bits=None, permutations=None):
        b = 200 #hash bits
        M = 30  #number of permutations
        self.data = data
        self.queries_dataset = queries_dataset
        self.b = hash_bits
        self.M = permutations
        self.model = LSH(X=data.X,b=self.b,M=self.M)
        self.L = None
        self.k = None
        self.query_features = None
        self.query_index = None

        self.sims = None
        self.sims_dec = None
        self.sims_ranked = None
        self.H_q = None
        self.hdist = None
        self.hdist_inc = None
        self.hdist_ranked = None
        self.ndcg_out = None
        
    def query_idx(self, query_index, N_neighbor_candidates=None, k_report=None):
        self.query_index = query_index
        self.L = N_neighbor_candidates
        self.k = k_report
        self.query_features = self.queries_dataset.get_features(query_index)

    def approx_top_k(self, refine):
        return self.model.approx_top_k(self.query_features, self.k, self.L, refine=refine)
                                       
    def random_top_k(self):
        return self.model.random_top_k(self.k)
    
    def randompp_top_k(self):
        return self.model.randompp_top_k(self.query_features, self.k, self.L)
    
    def exact_top_k(self):
        return self.model.exact_top_k(self.query_features, self.k)
    
    def ndcg(self, ranks):
        return  ndcg(self.sims,self.sims_dec,self.k,ranks)
    
    def get_distances(self):
        self.sims,self.sims_dec,self.sims_ranked = cos_sim_ordered(self.model.X, self.query_features)
       
        self.H_q = self.model._get_hash(self.query_features)
        self.hdist,self.hdist_inc,self.hdist_ranked = hamming_dist_ordered(self.model.H_data, self.H_q)
                
    def time_and_compare(self, N_queries, N_neighbor_candidates=None, k_report=None):
        times = np.zeros((N_queries))
        
        ndcg_0_opt      = np.zeros((N_queries))
        ndcg_1_hd       = np.zeros((N_queries))
        ndcg_2_lsh_hd   = np.zeros((N_queries))
        ndcg_3_lsh_ip   = np.zeros((N_queries))
        ndcg_4_random   = np.zeros((N_queries))
        ndcg_5_randompp = np.zeros((N_queries))
        
        for q_index in range(N_queries):            
            self.query_idx(q_index, N_neighbor_candidates, k_report)
            t1 = time.time()
            rank_3_lsh_ip   = self.approx_top_k(refine="innerprod")
            times[q_index] = time.time() - t1
            
            #Get Ranks
            self.get_distances()
            rank_0_opt      = self.sims_ranked[:self.k]
            rank_1_hd       = self.hdist_ranked[:self.k]
            rank_2_lsh_hd   = self.approx_top_k(refine="hamming")
            rank_4_random   = self.random_top_k()
            rank_5_randompp = self.randompp_top_k()
#             assert((self.exact_top_k() == rank_0_opt).all())

            #Compute NDCG Scores
            ndcg_0_opt[q_index]      = self.ndcg(rank_0_opt)
            ndcg_1_hd[q_index]       = self.ndcg(rank_1_hd)
            ndcg_2_lsh_hd[q_index]   = self.ndcg(rank_2_lsh_hd)
            ndcg_3_lsh_ip[q_index]   = self.ndcg(rank_3_lsh_ip)
            ndcg_4_random[q_index]   = self.ndcg(rank_4_random)
            ndcg_5_randompp[q_index] = self.ndcg(rank_5_randompp)
            

        self.ndcg_all   = [
            ndcg_4_random.mean(),
            ndcg_5_randompp.mean(),
            ndcg_1_hd.mean(),
            ndcg_2_lsh_hd.mean(),
            ndcg_3_lsh_ip.mean(),
            ndcg_0_opt.mean()
        ]
        self.ndcg_names = ["Random",
                           "Random w/ \nSimilarity Refinement",
                           "Linear Hamming",
                           "LSH w/ \nHamming Refinement",
                           "LSH w/ \nSimilarity Refinement",
                           "Linear Similarity"]
    
        return times.mean(), ndcg_3_lsh_ip.mean()
    
    def ndcg_analysis(self):
        assert self.sims is not None
        assert self.sims_dec is not None
        assert self.sims_ranked is not None
        assert self.hdist is not None
        assert self.hdist_inc is not None
        assert self.hdist_ranked is not None

        #Get Ranks
        rank_0_opt      = self.sims_ranked[:self.k]
        rank_1_hd       = self.hdist_ranked[:self.k]
        rank_2_lsh_hd   = self.approx_top_k(refine="hamming")
        rank_3_lsh_ip   = self.approx_top_k(refine="innerprod")
        rank_4_random   = self.random_top_k()
        rank_5_randompp = self.randompp_top_k()
        assert((self.exact_top_k() == rank_0_opt).all())

        #Compute NDCG Scores
        ndcg_0_opt      = self.ndcg(rank_0_opt)
        ndcg_1_hd       = self.ndcg(rank_1_hd)
        ndcg_2_lsh_hd   = self.ndcg(rank_2_lsh_hd)
        ndcg_3_lsh_ip   = self.ndcg(rank_3_lsh_ip)
        ndcg_4_random   = self.ndcg(rank_4_random)
        ndcg_5_randompp = self.ndcg(rank_5_randompp)

        self.ndcg_all   = [ndcg_4_random,ndcg_5_randompp,ndcg_1_hd,ndcg_2_lsh_hd,ndcg_3_lsh_ip,ndcg_0_opt]
        self.ndcg_names = ["Random","Random w/ \nSimilarity Refinement","Linear Hamming","LSH w/ \nHamming Refinement",
                      "LSH w/ \nSimilarity Refinement","Linear Similarity"]

    def ndgc_plot(self):
        plt.figure(figsize=(15,5))
        for i,(score,name) in enumerate(zip(self.ndcg_all,self.ndcg_names)):
            plt.bar([i],score)
            plt.text(i,score+0.01,str(round(score,3)),horizontalalignment="center")
        plt.xticks(np.arange(i+1),self.ndcg_names)
        plt.ylabel("Normalized Discounted Cumulative Gain")
        plt.show()
        
    def histograms(self):
        assert self.sims is not None
        #Distribution of Similarity Scores
        plt.hist(self.sims,bins=25,alpha=0.3,density=True,label="all")
        plt.legend()
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Probabililty Density")
        plt.show()
        
        #Distribution of Similarity Scores with overlay of LSH results
        rank_3_lsh_ip = self.approx_top_k(refine="innerprod")
        plt.hist(self.sims,bins=25,alpha=0.3,density=True,label="all")
        plt.hist(self.sims[rank_3_lsh_ip],alpha=0.3,density=True,label="lsh")
        plt.legend()
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Probabililty Density")
        plt.show()

        
    def sim_vs_ham(self):
        fig,ax = plt.subplots(1,1,figsize=(10,7))

        #Predicted relashionship with confidence interval
        plt.plot([0,1],[1,0],label="Predicted")
        xs = np.linspace(0,1,100)
        ci = binom_conf(self.model.b,xs,True)
        plt.fill_between(xs,ci[0],ci[1],alpha=0.2,label="95% Confidence Interval")
        del xs,ci

        #Actual observed relationship
        plt.scatter(self.sims,self.hdist,s=1,label="Actual")

        plt.xlabel("Cosine Similarity")
        plt.ylabel("Hamming Distance")
        plt.legend()
        plt.show()
        
        
    def plot(self, PLOTS_PER_QUERY=None):
        QUERIES_PER_PLOT = 1
        if PLOTS_PER_QUERY is None:
            PLOTS_PER_QUERY = self.k
        assert PLOTS_PER_QUERY <= self.k
        fig,ax = plt.subplots(2*QUERIES_PER_PLOT,1+PLOTS_PER_QUERY,sharex=True,sharey=True,figsize=(20,7))
        
        rank_0_opt = self.exact_top_k()
        rank_3_lsh_ip = self.approx_top_k(refine="innerprod")

        for i in range(QUERIES_PER_PLOT):
            base_row = 2*i

            #Query
            plt.sca(ax[base_row][0])
            self.queries_dataset.plt_img(self.query_index)

            #Turn off axis below query
            ax[base_row+1][0].axis('off')

            for j in range(PLOTS_PER_QUERY):
                #Exact
                plt.sca(ax[base_row][1+j])
                self.data.plt_img(rank_0_opt[j])

                #Approximate
                plt.sca(ax[base_row+1][1+j])
                self.data.plt_img(rank_3_lsh_ip[j])

        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.show()
        