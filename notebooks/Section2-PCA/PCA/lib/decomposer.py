import numpy as np
import os, sys

from pyspark.sql import Row

class Eigen_decomp:
    """A class for approximating a function with an orthonormal set of
    functions    """
    
    def __init__(self,x,f,mean,U):
        """ Initialize the widget

        :param x: defines the x locations
        :param f: the function to be approximated
        :param mean: The initial approximation (the mean)
        :param U: an orthonormal matrix with m columns (number of vectors to use in decomposition)
        :returns: None
        """

        self.U=U
        
        self.x=x
        self.mean=mean

        self.f=f
        self.startup_flag=True
        self.C=np.dot((np.nan_to_num(f-mean)),self.U)  # computer the eigen-vectors coefficients.
        self.C=np.array(self.C).flatten()
        self.m,self.n=np.shape(self.U)
        self.coeff={'c'+str(i):self.C[i] for i in range(self.C.shape[0])} # Put the coefficient in the dictionary format that is used by "interactive".
        return None

    def compute_var_explained(self):
        """Compute a summary of the decomposition

        :returns: ('total_energy',total_energy),
                ('residual var after mean, eig1,eig2,...',residual_var[0]/total_energy,residual_var[1:]/residual_var[0]),
                ('reduction in var for mean,eig1,eig2,...',percent_explained[0]/total_energy,percent_explained[1:]/residual[0]),
                ('eigen-vector coefficients',self.C)

        :rtype: tuple of pairs. The first element in each pair is a
        description, the second is a number or a list of numbers or an
        array.

        """
        def compute_var(vector):
            v=np.array(np.nan_to_num(vector),dtype=np.float64)
            return np.dot(v,v) # /float(total_variance)
        
        k=self.U.shape[1]
        residual_var=np.zeros(k+1)
        
        residual=self.f   # init residual to function 
        total_energy=compute_var(residual)
        residual=residual-self.mean # set residual to function - mean 
        residual_var[0]=compute_var(residual)
        # compute residuals after each 
        for i in range(k):
            g=self.U[:,i]*self.coeff['c'+str(i)]
            g=np.array(g).flatten()
            residual=residual-g # subtract projection on i'th coefficient from residual
            residual_var[i+1]=compute_var(residual)

        # normalize residuals
        _residuals=residual_var/(residual_var[0]+1e-10)   # Divide ressidulas by residuals after subtracting mean
        _residuals[0] = residual_var[0]/(total_energy+1e-10)

        return (('total_energy',total_energy),
                ('fraction residual var after mean, eig1,eig2,...',_residuals),
                ('eigen-vector coefficients',self.coeff))

    # total_var,residuals,reductions,coeff=recon.compute_var_explained()


def decompose_dataframe(sc,sqlContext,df,EigVec,Mean):
    """ run decompose(row) on all rows of a dataframe, return an augmented dataframe with columns
    corresponding to residuals and coefficients.
    """
    def decompose(row):
        """compute residual and coefficients for a single row      

        :param row: SparkSQL Row that contains the measurements for a particular station, year and measurement. 
        :returns: the input row with additional information from the eigen-decomposition.
        :rtype: SparkSQL Row 

        Note that Decompose is designed to run inside a spark "map()" command inside decompose_dataframe.
        Mean and EigVec are sent to the workers as global variables of "decompose"

        """
        #from numpy_pack import unpackAndScale
        import numpy_pack
        Series=np.array(numpy_pack.unpackAndScale(row),dtype=np.float64)
        #Mean=Mean_BC.value
        #EigVec=EigVec_BC.value
        recon=Eigen_decomp(None,Series,Mean,EigVec);
        total_var,residuals,coeff=recon.compute_var_explained()

        D=row.asDict()
        D['total_var']=float(total_var[1])
        D['res_mean']=float(residuals[1][0])
        for i in range(1,residuals[1].shape[0]):
            D['res_'+str(i)]=float(residuals[1][i])
            D['coeff_'+str(i)]=float(coeff[1]['c'+str(i-1)])
        return Row(**D)

    # main of decompose_dataframe
    
    #EigVec_BC=sc.broadcast(EigVec)
    #Mean_BC=sc.broadcast(Mean)
    import decomposer
    rdd2=df.rdd.map(decompose).cache()
    rdd2.count()
    return sqlContext.createDataFrame(rdd2)
