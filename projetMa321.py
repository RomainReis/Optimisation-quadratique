# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 15:48:25 2021

@author: Dos Reis
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import time


def gradientfix(A,b,X,tol,rho):
    """Fonction de la méthode du gradient à pas fixe"""
    
    i=1
    xit=[]
    xit.append(X)
    r=[]
    d=[]
    iMax=50000
    r.append(A@xit[0]-b)
    d.append(-r[0])
    xit.append(xit[0]+rho*d[0])
    while(np.linalg.norm((r[i-1]))>tol and i<iMax):
        r.append(A@xit[i]-b)
        d.append(-r[i])
        xit.append(xit[i]+rho*d[i])
        sol=xit[i]
        i=i+1
        nit=i
    return sol,xit,nit

def gradientoptimal(A,b,X,tol):
    """Fonction de la méthode du gradient à pas optimal"""
    
    i=1
    xit=[]
    xit.append(X)
    r=[]
    d=[]
    rho=[]
    iMax=50000
    r.append(A@xit[0]-b)
    d.append(-r[0])
    rho.append((np.transpose(r[0])@r[0])/(np.transpose(r[0])@A@r[0]))
    xit.append(xit[0]+rho[0]*d[0])
    while(np.linalg.norm((r[i-1]))>tol and i<iMax):
        r.append(A@xit[i]-b)
        d.append(-r[i])
        rho.append((np.transpose(r[i])@r[i])/(np.transpose(r[i])@A@r[i]))
        xit.append(xit[i]+rho[i]*d[i])
        sol=xit[i]
        i=i+1
        nit=i
    return sol,xit,nit

def gradientconj(A,b,X,tol):
    """Fonction de la méthode du gradient à pas conjugué"""
    
    i=1
    xit=[]
    xit.append(X)
    r=[]
    d=[]
    rho=[]
    beta=[]
    iMax=50000
    r.append(A@xit[0]-b)
    d.append(-r[0])
    rho.append((np.transpose(r[0])@r[0])/(np.transpose(d[0])@A@d[0]))
    xit.append(xit[0]+rho[0]*d[0])
    
    while(np.linalg.norm((r[i-1]))>tol and i<iMax):            
        r.append(A@xit[i]-b)
        beta.append((np.linalg.norm(r[i])**2)/(np.linalg.norm(r[i-1])**2))
        d.append(-r[i]+beta[i-1]*d[i-1])
        rho.append((np.transpose(r[i])@r[i])/(np.transpose(d[i])@A@d[i]))
        xit.append(xit[i]+rho[i]*d[i])
        sol=xit[i]
        i=i+1
        nit=i
        
    return sol,xit,nit


def Tendance(x, a, b):
    """fonction qui permet d'afficher une courbe de tendance"""

    return a*x+b

######Lecture des données######

p=np.loadtxt("./dataP.dat")
q=np.loadtxt("./dataQ.dat")


######Nuage de points######
plt.title("Courbe de la taille Q en fonction de l'age P")
plt.grid(True)
plt.scatter(p,q, marker="x")

params = [0, 2]  

params, covariance = optimize.curve_fit(Tendance, p, q, params)
plt.plot(p, Tendance(p, *params), "-b", color="red")

plt.xlabel('Age des enfants')
plt.ylabel('Taille des enfants')
plt.show()

########Creation des Matrices et vecteurs du probleme#########

p.resize((50,1))
u=np.ones((50,1))

a=np.hstack([u,p])#la matrice X appelée ici a
A=np.transpose(a)@a#la matrice A=XtX
b=np.transpose(a)@q


######Determination de c* grace a numpy pour les tracés de courbes########

cstar=np.linalg.solve(A,b)#on utilise cette fonction de numpy car on est pas censé avoir réalisé les algorythmes a ce niveau
#du TP
c1=cstar[0]
c2=cstar[1] #on associe les valeurs de c* aux variables c1 et c2 pour alleger les ecritures


##########calcul des couples propres##################

lambda1,lambda2=np.linalg.eigvals(A)
print("\nLes valeurs propores de la matrice A=XtX sont : ",lambda1," et ",lambda2)
v,w=np.linalg.eig(A)
cond=np.linalg.cond(A)
v1=w[:,0]
v2=w[:,1]
print("Les vecteurs propres de la matrice A=XtX sont: ",v1," et ",v2)
print("\non a les couples propres suivant:")
print("\ncouple 1: (",lambda1,",",v1,")")
print("couple 2: (",lambda2,",",v2,")")
print("\nLe conditionnement de la matrice A=XtX est: ",cond)

##########calcul et tracé des fonctions partielles#############

s=np.transpose(q)@q
Zstar = 1/2*((A[0,0]*(c1**2))+(2*A[0,1]*c1*c2)+(A[1,1]*(c2**2))-2*((b[0]*c1)+(b[1]*c2))+s)
#Zstar est la valeur de la fonction f evaluée en C* cette valeur est necessaire pour le calcul
#des fonctions partielles

e1=np.array([0,1])#element de la base canonique de R²
e2=np.array([1,0])#element de la base canonique de R²

F=[]
T=[]
for t in range (-10,11):
    F.append(0.5*np.transpose(e1)@A@e1*(t**2)-np.transpose(b-A@cstar)@e1*t+Zstar)
    T.append(t)

    

fig, axs = plt.subplots(2, 2,figsize=(15,10))
axs[0, 0].plot(T, F)
axs[0, 0].set_title('Fonction partielle de F en c* suivant e1')
axs[0, 0].grid(True) 

F=[]
T=[]
for t in range (-10,11):
    F.append(0.5*np.transpose(e2)@A@e2*(t**2)-np.transpose(b-A@cstar)@e2*t+Zstar)
    T.append(t)
    
axs[0, 1].plot(T, F)
axs[0, 1].set_title('Fonction partielle de F en c* suivant e2')
axs[0, 1].grid(True) 

F=[]
T=[]
for t in range (-10,11):
    F.append(0.5*np.transpose(v1)@A@v1*(t**2)-np.transpose(b-A@cstar)@v1*t+Zstar)
    T.append(t)

axs[1, 0].plot(T, F, 'tab:orange')
axs[1, 0].set_title('Fonction partielle de F en c* suivant v1')
axs[1, 0].grid(True) 

F=[]
T=[]
for t in range (-10,11):
    F.append(0.5*np.transpose(v2)@A@v2*(t**2)-np.transpose(b-A@cstar)@v2*t+Zstar)
    T.append(t)
    
axs[1, 1].plot(T, F, 'tab:orange')
axs[1, 1].set_title('Fonction partielle de F en c* suivant v2')
axs[1, 1].grid(True)     


plt.show()


######Representations geometriques de la fonction F######

c1 = np.linspace(-10,10,40)
c2 = np.linspace(-10,10,40)


X,Y = np.meshgrid(c1,c2)
Z = 1/2*((A[0,0]*(X**2))+(2*A[0,1]*X*Y)+(A[1,1]*(Y**2))-2*((b[0]*X)+(b[1]*Y))+s)#equation de la fonction F
dx,dy=np.gradient(Z)


############ Tracé des lignes de niveau##############

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
ax.set_title('Tracé des lignes de niveau')
ax.set_xlabel('X')
ax.set_ylabel('Y')

plt.show()

############ Lignes de niveau plus précis##############

fig, ax = plt.subplots()
CS1 = ax.contour(X, Y, Z,levels=[200,500,1000,5000,10000,20000,50000,100000])
ax.set_title('Tracé des lignes de niveau')
ax.set_xlabel('X')
ax.set_ylabel('Y')

plt.show()
############ Tracé de la Surface de F############

fig = plt.figure(figsize=(8,6))
ax3d = plt.axes(projection='3d')

ax3d = plt.axes(projection='3d')
ax3d.plot_surface(X,Y,Z,cmap='plasma')
ax3d.set_title('Courbe de la surface de F')
ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('Z')

plt.show()

###############Calcul de c* en fonction de p,q et 1################################
U=np.ones((50,1))#creation du vecteur unitaire
c1star=((np.linalg.norm(p)**2)*np.transpose(q)@U-(np.transpose(p)@U)*(np.transpose(p)@q))/(50*(np.linalg.norm(p)**2)-(np.transpose(p)@U)**2)
c2star=(50*(np.transpose(p)@q)-(np.transpose(q)@U)*(np.transpose(p)@U))/(50*(np.linalg.norm(p)**2)-(np.transpose(p)@U)**2)
print("\ncalcul de c* a l'aide de p,q et 1 :")
print("c1*: ",c1star)
print("c2*: ",c2star)

###############test des methodes pour la resolution du probleme####################

tol=10**-6
X=np.array([-9,-7])
rho=0.001
tmps1=time.perf_counter ()
solGPF,xitGPF,nitGPF=gradientfix(A,b,X,tol,rho)
tmps2=time.perf_counter ()
print("\nSolution gradient à pas fixe : ",solGPF,", nombre d'itérations:",nitGPF)
print("temps d'exécution de la méthode à pas fixe : ",tmps2-tmps1, "sec")

tmps3=time.perf_counter ()
solGPO,xitGPO,nitGPO=gradientoptimal(A,b,X,tol)
tmps4=time.perf_counter ()
print("\nSolution gradient à pas optimal: ",solGPO,", nombre d'itérations:",nitGPO)
print("temps d'exécution de la méthode à pas optimal : ",tmps4-tmps3,"sec")

tmps5=time.perf_counter ()
solGC,xitGC,nitGC=gradientconj(A,b,X,tol)
tmps6=time.perf_counter ()
print("\nSolution gradient conjugué : ",solGC,", nombre d'itérations:",nitGC)
print("temps d'exécution de la méthode à pas conjugué : ",tmps6-tmps5,"sec")

################## Tracé du quiver #####################

fig, ax = plt.subplots()
CS = ax.quiver(c1, c2, dx,dy,cmap='ocean')
ax.set_title('visualisation du gradient')
ax.set_xlabel('X')
ax.set_ylabel('Y')

plt.show()

##################Visualisation de la covergeance########################
#################sur les lignes de niveaux###########################
c1 = np.linspace(-10,10,40)
c2 = np.linspace(-10,10,40)


X,Y = np.meshgrid(c1,c2)
Z = 1/2*((A[0,0]*(X**2))+(2*A[0,1]*X*Y)+(A[1,1]*(Y**2))-2*((b[0]*X)+(b[1]*Y))+s)#equation de la fonction F
dx,dy=np.gradient(Z)


############ Tracé des lignes de niveau et de la trajectoire reliant les iterations a la solution##############
########pour le gradient a pas fixe######################
fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
ax.set_title('Tracé des lignes de niveau et de la trajectoire des itérations')
ax.set_xlabel('X')
ax.set_ylabel('Y')
xs = [x[0] for x in xitGPF]
ys = [x[1] for x in xitGPF]
plt.plot(xs, ys,'--r')
plt.show()


############ Tracé des lignes de niveau et de la trajectoire reliant les iterations a la solution##############
###########pour le gradient a pas optimal##############
fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
ax.set_title('Tracé des lignes de niveau et de la trajectoire des itérations')
ax.set_xlabel('X')
ax.set_ylabel('Y')
xs = [x[0] for x in xitGPO]
ys = [x[1] for x in xitGPO]
plt.plot(xs, ys,'--r')
plt.show()

############ Tracé des lignes de niveau et de la trajectoire reliant les iterations a la solution##############
##########pour le gradient conjugué##############
fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
ax.set_title('Tracé des lignes de niveau et de la trajectoire des itérations')
ax.set_xlabel('X')
ax.set_ylabel('Y')
xs = [x[0] for x in xitGC]
ys = [x[1] for x in xitGC]
plt.plot(xs, ys,'--r')
plt.show()

##################Visualisation des modeles calculés#####################
plt.grid(True)
plt.scatter(p,q, marker="x")
plt.plot([2.0, 8.0],[solGPF[1]*2+solGPF[0],solGPF[1]*8+solGPF[0]],'r')
plt.plot([2.0, 8.0],[solGPO[1]*2+solGPO[0],solGPO[1]*8+solGPO[0]],'b')
plt.plot([2.0, 8.0],[solGC[1]*2+solGC[0],solGC[1]*8+solGC[0]],'g')
plt.xlabel('Age des enfants')
plt.ylabel('Taille des enfants')
plt.title("Courbe de la taille Q en fonction de l'age P")
plt.legend(['pas fixe', 'pas optimal', 'pas conjugué'])
plt.show()




