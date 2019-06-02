import numpy as np
import copy
import time
import pdb

class LSH:
    def __init__(self, X, b, M, B):
        """
        Initialize an LSH class with a dataset X
        and hyperparameters b and M.
        @param X: np.ndarray shape = (d,n)
            The data set over which we want to perform
            nearest neigbor queries. Each column is a 
            d-dimensional vector.
        @param b: int
            The number of bits in our hash keys. Larger
            b leads to higher qualilty matches, but slower
            query times.
        @param m: int
            Number of permutations of the bit vectors to
            use for nearest neighbor searches. Larger M
            leads to higher quality matches, but slower
            query times.
        """
        # Check Arguments
        self.b = int(b)
        self.M = int(M)
        self.B = int(B/2)
        assert(self.b >= 10 and self.b <= 400)
        assert(self.M >= 1  and self.M <= 100)
        assert(self.B >= 0  and self.B <=300)
        
        # Store local copy of inputs vectors : (d,n)
        # Normalize data as cosine similarity is invariant
        # to scaling. We can then treat this as max inner 
        # product
        self.X = X
        #self.X = copy.deepcopy(X)
        self.X = self.X/np.linalg.norm(self.X,keepdims=True,axis=0)
        self.d, self.n = self.X.shape
        
        # Generate Random Hash Vectors : (b,d)
        # Each row is a vector
        self.R = np.random.normal(0,1,(self.b,self.d)) 

        # Permuations : (M,b)
        # Each row is a permutation of indices 0,1,...,b-1
        Sigma = []
        for _ in range(self.M):
            Sigma.append(np.random.permutation(self.b))
        self.Sigma = np.array(Sigma)
        
        # Hash the dataset : (b,n)
        self.H_data = self._get_hash(self.X)
        
        # Permute the hashes : (M,b,n)
        self.H_data_permute = self._permute(self.H_data)
        
        # Convert bit-hash to number for search : (M,n)
        self.V_data = self._bitstr2float(self.H_data_permute)
        
        # Sort the numeric values to allow for binary search
        # Need to record the index of the original values in the
        # sorted order
        self.V_data_sort_idx = np.argsort(self.V_data,axis=-1) # (M,n)
        self.V_data_sort_val = np.sort(self.V_data,axis=-1)    # (M,n)
        i = int(self.M/2)
        j = int(self.n/3)
        assert(self.V_data[i,self.V_data_sort_idx[i][j]] == self.V_data_sort_val[i,j])
        
        # Store offset column for use in queries
        self.offt = np.array([np.arange(-self.B,self.B+1,1)]).T
    
    # Hash of Data : (b,n)
    def _get_hash(self,X):
        """
        Hashes a dataset X of d-dimensional Euclidean
        vectors to b-dimensional bit vectors. Hashing
        is based on projections of the data on to random
        vectors and rounding.
        @param X: np.ndarray shape = (d,n)
            The data set to hash. Each column is a 
            d-dimensional vector to hash
        @returns: np.ndarray shape = (b,n)
            Each column is a b-bit hash of the
            corresponding column in X
        """
        assert(len(X.shape)==2)
        d,_ = X.shape
        assert(d == self.d)
        tmp = self.R@X #matrix mult is inner products
        H = np.zeros(tmp.shape,dtype=np.int8) # 1 byte
        H[tmp>=0] = 1
        del tmp
        return H
    
    # Permute Rows of Hash: (M,b,n)
    def _permute(self,H):
        """
        Takes a hash matrix and produces permutations
        of the hash keys based on the sorted orders in 
        self.Sigma.
        @param H: np.ndarray shape = (b,m)
            Bit matrix
        @returns: np.ndarray shape = (M,b,m)
            Permuation of the input H based on the
            permutation matrix Sigma. The permutation
            matrix specifies the order in which to
            permute the rows of H. There are M seperate,
            independent, permutations to output.
        """
        assert(H.shape[0] == self.b)
        H_permute = []
        for i in range(self.M):
             H_permute.append(H[self.Sigma[i],:]) #permute rows
        H_permute = np.array(H_permute)
        assert(H_permute[0].sum() == H_permute[-1].sum())
        return H_permute
    
    # Interpret axis 1 (middle) as bit string: (M,n)
    def _bitstr2float(self,H_permute):
        """
        Takes various permutations of a
        hash key matrix and converts individual
        hash keys (b-bit string) to floating
        point numbers. We use floats because numpy doesn't
        support infinite precision integers. This conversion
        allows us to do a binary search over the hashkeys
        without checking each bit individually.
        @param H_permute: nd.array shape = (M,b,n)
        """
        assert(len(H_permute.shape) == 3)
        b = H_permute.shape[1]
        
        #column of positional values
        pos = np.exp2(np.arange(b-1,-1,-1)).reshape(1,-1,1)
        
        #broadcast product then sum
        V = (H_permute*pos).sum(axis=1)
        return V
          
    def approx_top_k(self,q,k,refine="hamming"):
        """
        Returns the top-k matches in the dataset
        for a query q.
        @param q: np.ndarray shape = (d,m)
            Each column is a d-dimensional vector
            for which we want the k-nearest neighbors
        @param k: int
            The number of neighbors to return
        @return: np.adarray (m,k)
            Index of top-k matches for each of the
            m queries. The indices index into the 
            original data
            
        """
        assert(len(q.shape)==2)
        d,m = q.shape
        assert(d == self.d)
        assert(refine in ["hamming","innerprod"])
        
        # Hash the query : (b,m) 
        H_q = self._get_hash(q)
        
        # Permute the query : (M,b,m)
        H_q_permute = self._permute(H_q)
        
        # Convert bit-hash to number for search : (M,m)
        V_q = self._bitstr2float(H_q_permute)

        # Get candidates : list of sets
        candidates = [set() for _ in range(m)]
        #explicit loop through sorted orders, parallel processing of queries
        for i in range(self.M):
            #Binary Search : Once per sorted order, for all queries in parallel : (m,)
            #Binary search for the place of m items in list of n things
            val_idx_0 = np.searchsorted(self.V_data_sort_val[i],V_q[i])
            
            # Include candidates near the bin-search match : (B,m) 
            # Each column is the matches for a query
            val_idx = np.repeat(self.offt,m,axis=-1) + val_idx_0 #broadcast the row
            val_idx = np.maximum(val_idx,0)
            val_idx = np.minimum(val_idx,self.n-1)
            
            # Get true indices into original dataset : (B,m)
            idx = self.V_data_sort_idx[i,val_idx]
            
            #Sanity check indirection works properly
            assert((self.V_data[i,idx] == self.V_data_sort_val[i,val_idx]).all())

            #Add matches to set
            for j in range(m):
                candidates[j] |= set(idx[:,j])
        
        # Get top-k
        ans = [None for _ in range(m)]
        for i in range(m):
            c = np.array(list(candidates[i])) #indices into X
            # pdb.set_trace()
            if refine == "hamming":
                print(self.H_data.shape, c.shape)
                # t1 = time.time()
                zz = self.H_data[:,c]
                # t2 = time.time()
                hd = np.sum(zz != H_q,axis=0) #hamming distance
                hd_argsort = np.argsort(hd)[:k] #sort increasing
                top_k = c[hd_argsort]
                # t3 = time.time()
                # print(t2-t1, t3-t2)
            elif refine == "innerprod":
                print(self.X.shape, c.shape)
                # t1 = time.time()
                zz = self.X[:,c]
                # t2 = time.time()
                ip = (q[:,i:i+1].T@zz)[0] #inner products
                ip_argsort = np.argsort(ip)[::-1][:k] #sort decreasing
                top_k = c[ip_argsort]
                # t3 = time.time()
                # print(t2-t1, t3-t2)
            else:
                raise ValueError("Refinement method must be based on Hamming Distance or Inner Product.")
            ans[i] = top_k
        ans = np.array(ans)
       
        if m == 1:
            return ans[0]
        return ans
    
    def exact_top_k(self,q,k):
        ip = (q.T@self.X)[0] #inner products
        ip_argsort = np.argsort(ip)[::-1][:k] #sort decreasing
        return ip_argsort
    