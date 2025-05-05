from numpy import *
from numpy.random import choice
class KD_tree:
    """A class that represents the whole KDtree,
    Points to the root KD_node"""
    def __init__(self,data,limit=100,depth=8):
        """ Instantiate a KDtree:
        data = training data each row is an example, the number of columns is the dimension.
        limit,depth  = nodes are split into two children only if their depth is smaller than depth 
                       and the number of examples in the node is at least limit"""
        self.data_size=data.shape[0]
        self.root=KD_node(self,data,limit=limit,depth=depth,path=[])
    def calc_encoding(self,data):
        """calculate a log ratio encoding for a new set of vectors (=image)"""
        data_size=data.shape[0]
        return self.root.calc_encoding(data,data_size)

class KD_node:
    """ the main class in the implementation of KD-tree, encodes a single node in the tree"""
    def __init__(self,tree,data,limit=100,depth=8,path=[]):
        #print(len(path))
        self.tree=tree
        self.path=path
        self.read_path=''.join([str(x) for x in path])
        self.size,self.dim=data.shape
        self.prob=data.shape[0]/self.tree.data_size
        #print('%10s  %3.3g'%(self.read_path,self.prob))
 
        if self.size<limit or len(path)>depth:
            self.leaf=True
        else:
            self.leaf=False
            index=random.choice(self.dim)
            H=data[:,index]
            threshold=median(H)
            below=data[data[:,index]<threshold,:]
            above=data[data[:,index]>=threshold,:]
            self.threshold=threshold
            self.index=index
            self.above=KD_node(tree,above,path=self.path+[1],depth=depth)
            self.below=KD_node(tree,below,path=self.path+[0],depth=depth)

    def calc_encoding(self,data,full_data_size,limit=100,smooth=1e-7):
        """Use trained tree to encode an individual dataset (image)"""
        my_prob=data.shape[0]/full_data_size
        log_ratio=log((my_prob+smooth)/(self.prob+smooth))
        my_result=[(self.read_path,log_ratio)]
        if self.leaf or data.shape[0] < limit:
            return my_result
        else:
            below=data[data[:,self.index]<self.threshold,:]
            above=data[data[:,self.index]>=self.threshold,:]
            above_results=self.above.calc_encoding(above,full_data_size,limit=limit)
            below_results=self.below.calc_encoding(below,full_data_size,limit=limit)
            return my_result+above_results+below_results

    def calc_density(self,data):
        """Calculate density in box defined by node"""
        if(data.shape[0]<2):
            self.density=0
            return 0
        bounding_box={i:(min(data[:,i]),max(data[:,i])) for i in range(self.dim)}
        Vol=1
        for i in range(self.dim):
            _min,_max=bounding_box[i]
            Vol*=(_max-_min)
        self.density=data.shape[0]/(Vol+0.001)
        return self.density
    
def train_encoder(files,max_images=200,tree_depth=8):
    """Train an encoding tree using a set of images
    If there are more than man_images image, choose max_images from them 
    by selecting at random w/o replacement"""
    ## Collect data for training
    _len=len(files)
    if _len<=max_images:    # if more than max_images files, sample max_images without replacement
        selected_files=files
    else:
        I = choice(range(_len),max_images,replace=False)
        selected_files=[files[i] for i in I]
    print('used %d images to train KDTree'%len(selected_files))

    Plist=[]
    for  i in range(len(selected_files)):
        M=load(selected_files[i])
        Image=M['x']
        pixels=Image.reshape((Image.shape[0], -1)).T
        Plist.append(pixels)
    data=concatenate(Plist,axis=0)

    print('KDTree training data shape=',data.shape)
    
    ## train tree
    train_size=data.shape[0]
    tree=KD_tree(data,depth=tree_depth)
    return train_size,tree

def encode_image(file,tree):
    M=load(file)
    Image=M['x']
    pixels=Image.reshape((Image.shape[0], -1)).T
    code=tree.calc_encoding(pixels)
    return code

class encoded_dataset:

    def __init__(self,image_dir,df,tree,depth=8,label_col='rich'):

        def bin2int(c):
            ans=0
            for j in range(len(c)):
                bit=int(c[-(j+1)])
                ans+=bit*(2**j)
            return ans
  
        self.df=df
        self.rows=df.shape[0]
        self.cols=2**(depth+1)+1
        data=zeros([self.rows,self.cols]) #code length +1 for label

        j=0
        for filename,row in df.iterrows():
            filepath=f"{image_dir}/{filename}"
            code = encode_image(filepath,tree)

            V=zeros(self.cols-1)
            for c,a in code:
                V[bin2int(c)]=a

            label=row[label_col]*1

            data[j,-1]=label
            data[j,:-1]=V

            if((j+1) %10==0):
                print(j,filename,end='\r')
            j+=1

        self.data=data

    def get_df(self):
        return self.df

    def get_slice(self,selection):
        assert selection.shape[0] == self.data.shape[0]
        S=data[selection,:]
        return S
        

