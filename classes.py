import numpy as np
class deform_grad:
    def __init__(self, F=None, dF=None, A=None, F_r=None, m=None) :
         # F - deformation gradient
         self.F = F
         # dF - derivative of the deformation gradient with respect to nodal displacements displacements
         # dF = [dF/du_1; dF/dv_1; dF/du_2; dF/dv_2; dF/du_3; dF/dv_3;
         # dF/du_i and dF/dv_i are tensors written as columns:
         # dF/du_i=[dF/du_i(1,1) dF/du_i(2,1) dF/du_i(1,2) dF/du_i(2,2)];
         self.dF = dF
         # A - areas
         self.A = A
         # F_r=F*m - reduced matrix after the Lagrange reduction
         self.F_r = F_r
         # m - corresponding transform
         self.m = m
    def Reduction(self):
        e1=np.copy(self.F[:,0])
        e2=np.copy(self.F[:,1])
        if (np.linalg.norm(e2)>np.linalg.norm(e1)):
            ### modify vector e2
            while (np.linalg.norm(e2)>np.linalg.norm(e2-e1)) | (np.linalg.norm(e2)>np.linalg.norm(e2+e1)):
                if (np.linalg.norm(e2-e1)<np.linalg.norm(e2+e1)):
                    e2=(e2-e1)
                else:
                    e2=e2+e1
            ### modify vector e1
            while (np.linalg.norm(e1)>np.linalg.norm(e1-e2)) | (np.linalg.norm(e1)>np.linalg.norm(e2+e1)):
                if (np.linalg.norm(e1-e2)<np.linalg.norm(e2+e1)):
                    e1=(e1-e2)
                else:
                    e1=e2+e1
        else:
           ### modify vector e1
           while (np.linalg.norm(e1)>np.linalg.norm(e1-e2)) | (np.linalg.norm(e1)>np.linalg.norm(e2+e1)):
               if (np.linalg.norm(e1-e2)<np.linalg.norm(e2+e1)):
                   e1=(e1-e2)
               else:
                   e1=e2+e1
           ### modify vector e2
           while (np.linalg.norm(e2)>np.linalg.norm(e2-e1)) | (np.linalg.norm(e2)>np.linalg.norm(e2+e1)):
               if (np.linalg.norm(e2-e1)<np.linalg.norm(e2+e1)):
                   e2=(e2-e1)
               else:
                   e2=e2+e1
        F_r=np.array([[e1[0],e2[0]],[e1[1],e2[1]]])
        m=np.linalg.inv(np.copy(self.F)).dot(F_r)
        return(m.astype(int))
    def well(self):
        e1=np.copy(self.F[:,0])
        e2=np.copy(self.F[:,1])
        ## control parameters
        c_1=1
        c_2=0
        w=0
        while (c_1 | c_2):
            c_1=1
            c_2=0
            if (np.linalg.norm(e1)>np.linalg.norm(e2)):
                e1_copy=np.copy(e1)
                e1=e2
                e2=e1_copy
                c_2=1
            if np.dot(e1,e2)<0:
                e2=-e2
                c_2=1
            de=e1-e2
            if (np.linalg.norm(de)<np.linalg.norm(e2)):
                e2=de
                w+=1
                c_2=1
            c_1=0
        return w
            
                

class en_density:
    def __init__(self, W=None, dW=None) :
         # W - value of the energy density
         self.W = W
         # dW - derivative with respect to the metric tensor C
         self.dW = dW
class en_parameters:
    def __init__(self, en_type=None, K=None, C=None) :
         # type of the energy density = 'kirchhoff', 'poly'
         self.en_type = en_type
         # stifness matrix C (for 'kirchhoff')
         # en_parameters.C='sq' or 'hex' for polynomial energy (square and hexagonal symmetry, respectively)
         self.C = C
         # bulk modulus
         self.K = K
class gl_energy:
    def __init__(self, E=None, dE=None, E_el=None, stress_val=None, U=None) :
         # global energy of a specimen
         # its value
         self.E = E
         # derivatives with respect to displacements U
         self.dE = dE
         # energy of each element
         self.E_el = E_el
         # class stress (see in the same file class 'stress')
         self.stress_val = stress_val
         # displacements
         self.U = U
class stress:
    def __init__(self, sigma=None, piola_1=None, piola_2=None, sigma_tot=None, piola_1_tot=None, piola_2_tot=None):
         # stresses in elements and total values
         # Cauchy stresses in each element
         self.sigma = sigma
         # 1 Piola-Kirchhoff stresses in each element
         self.piola_1 = piola_1
         # 2 Piola-Kirchhoff stresses in each element
         self.piola_2 = piola_2
         # Total Cauchy stresses
         self.sigma_tot = sigma_tot
         # Total 1 Piola-Kirchhoff stresses
         self.piola_1_tot = piola_1_tot
         # Total 2 Piola-Kirchhoff stresses
         self.piola_2_tot = piola_2_tot
class stress_element:
    def __init__(self, sigma=None, piola_1=None, piola_2=None):
         # stresses in elements and total values
         # Cauchy stresses in each element
         self.sigma = sigma
         # 1 Piola-Kirchhoff stresses in each element
         self.piola_1 = piola_1
         # 2 Piola-Kirchhoff stresses in each element
         self.piola_2 = piola_2
class displ_bc:
    def __init__(self, top=None, bottom=None, left=None, right=None) :
         # displacement on the TOP boundary
         self.top = top
         # displacement on the BOTTOM boundary
         self.bottom = bottom
         # displacement on the LEFT boundary
         self.left = left
         # displacement on the RIGHT boundary
         self.right = right        
class BCs:
    #### the displacements u0 and v0 should be of the class displ_bc
    def __init__(self, top=None, bottom=None, left=None, right=None, u0=None, v0=None) :
         ####### nodes (indices) of the boundaries
         # TOP boundary
         self.top = top
         # BOTTOM boundary
         self.bottom = bottom
         # LEFT boundary
         self.left = left
         # RIGHT boundary
         self.right = right
         
         ####### HORIZONTAL displacements of the boundaries
         self.u0 = u0
         
         ####### VERTICAL displacements of the boundaries
         self.v0 = v0
class bulk_node:
    #### prescribed displacements on one of the bulk nodes
    def __init__(self, node=None, u0=None, v0=None, free = None) :
        ### indices of the nodes
        self.node=node
        ### HORIZONTAL displacements
        self.u0=u0
        ### VERTICAL displacements
        self.v0=v0
        ### free nodes
        self.free=free
class geom_gen:
    #### generate the geometry from points p and connectivity list T
    def __init__(self, p=None, T=None):
        # nodes
        self.p=np.array(p)
        # Connectivity matrix
        
        self.T=T
        
        # TOP boundary
        p_top=np.where(np.abs(p[:,1]-np.max(p[:,1]))<0.01)
        # BOTTOM boundary
        p_bottom=np.where(np.abs(p[:,1]-np.min(p[:,1]))<0.01)
        # LEFT boundary
        p_left=np.where(np.abs(p[:,0]-np.min(p[:,0]))<0.01)
        # RIGHT boundary
        p_right=np.where(np.abs(p[:,0]-np.max(p[:,0]))<0.01)
        # ALL boundary nodes
        p_bound=np.unique(np.concatenate((p_bottom,p_top,p_left,p_right)))
        p_bulk=np.setdiff1d(range(p.shape[0]), p_bound)
        
        # initialise the boundary and bulk nodes
        self.top=np.array(p_top).T
        self.bottom=np.array(p_bottom).T
        self.left=np.array(p_left).T
        self.right=np.array(p_right).T
        self.bound=np.array(p_bound).T
        self.bulk=np.array(p_bulk).T
