import numpy as np
import math as ma
import numpy.random as rand
#Erstelle POSCAR-Datei fuer beliebig grosse Superzelle
#Benoetigt Datei 'relativkoord-MA.txt' mit Relativkoordinaten 
#in direct-Format

f=open("POSCAR_MA_volume","w")
fkoord=open("relativkoord-MA.txt","r")

a=float(1.0)   #Faktor
aG=float(6.07) #Gitterkonstante; Ursprünglicher Wert: 6.32
atoms="Pb C N H I\n"  #Atomsorten

#Array um Koordinaten zwischen zu speichern
Pb_list=np.empty([0,3])
C_list=np.empty([0,3])
N_list=np.empty([0,3])
H_list=np.empty([0,3])
I_list=np.empty([0,3])

koord=np.empty([0,3]) #Array aller endgueltigen Koordinaten
x=np.array([0,0,0])

mode=int(input('Eingabemodus Winkel:\n 1=per Hand\n 2=Zufallszahl\n 3=Winkel 000\n 4=Datei Zufallswinkel\n'))
if mode==2:
    fwinkelRand=open("Zufallswinkel","w")
    fwinkelRand.write('%4s %4s %4s %8s %8s %8s\n' %('x','y','z',\
    'PHI','THETA','PSI'))
if mode==4:
    fZufWinkel=open("Zufallswinkel","r")
    fZufWinkel.readline()
    
    
xAnz=int(input('Anzahl EZ in x-Richtung '))
yAnz=int(input('Anzahl EZ in y-Richtung '))
zAnz=int(input('Anzahl EZ in z-Richtung '))

anz=xAnz*yAnz*zAnz #Anzahl an Elementarzellen
#Anzahl der einzelnen Atome
anzahl=np.array([anz,anz,anz,6*anz,3*anz]) 

#Basisvektoren
a1=np.array([aG*xAnz,0,0])
a2=np.array([0,aG*yAnz,0])
a3=np.array([0,0,aG*zAnz])

#lese Relativkoordinaten des Molekuels ein
fkoord.readline()
temp1=fkoord.readline().split()
temp2=fkoord.readline().split()
temp3=fkoord.readline().split()
temp4=fkoord.readline().split()
temp5=fkoord.readline().split()
temp6=fkoord.readline().split()
temp7=fkoord.readline().split()
temp8=fkoord.readline().split()

#Direct-Koordinaten der Atome,
#wobei Molekuel noch am Ursprung ist 
Pb1=np.array([0,0,0]) 
C1=np.array([float(i) for i in temp1])
N1=np.array([float(i) for i in temp2])
H1=np.array([float(i) for i in temp3])
H2=np.array([float(i) for i in temp4])
H3=np.array([float(i) for i in temp5])
H4=np.array([float(i) for i in temp6])
H5=np.array([float(i) for i in temp7])
H6=np.array([float(i) for i in temp8])
I1=np.array([0.5,0,0])
I2=np.array([0,0.5,0])
I3=np.array([0,0,0.5])
#Array aller Atomkoordinaten
b=np.array([Pb1,C1,N1,H1,H2,H3,H4,H5,H6,I1,I2,I3])

#Rotation des Molekuels um Koordinatenursprung
def rotation(phi, theta, psi, vector):  
    #Wandle winkel in radialen Winkel um 
    phi=ma.radians(phi)
    theta=ma.radians(theta)
    psi=ma.radians(psi)
    #cos(Winkel)
    cphi=ma.cos(phi)
    ctheta=ma.cos(theta)
    cpsi=ma.cos(psi)
    #sin(Winkel)
    sphi=ma.sin(phi)
    stheta=ma.sin(theta)
    spsi=ma.sin(psi)
    #Drehung um z-Achse
    rotZ=np.matrix([[cphi,-sphi,0],[sphi,cphi,0],[0,0,1]])
    #Drehung um y-Achse
    rotY=np.matrix([[ctheta,0,stheta],[0,1,0],[-stheta,0,ctheta]]) 
    #Drehung um x-Achse
    rotX=np.matrix([[1,0,0],[0,cpsi,-spsi],[0,spsi,cpsi]])
    #Drehmatrix rotZ*rotY*rotX
    rot=np.dot(rotZ,rotY)
    rot=np.dot(rot,rotX)
    temp=np.matmul(vector,rot)#gedrehter Vektor
    return np.array([temp[0,0],temp[0,1],temp[0,2]])
    
#Speichern der einzelnen Atomkoordinaten 
def speichern(k,y): 
    #speichere Atomkoordinaten je nach Sorte in passender 
    #Liste
    if k==0: #Pb-Atoom
        global Pb_list
        Pb_list=np.append(Pb_list,[y],axis=0)
    if k==1: #C-Atom
        global C_list
        C_list=np.append(C_list,[y],axis=0)
    if k==2: #N-Atom
        global N_list
        N_list=np.append(N_list,[y],axis=0)
    if (k>2 and k<9): #H-Atom
        global H_list
        H_list=np.append(H_list,[y],axis=0)
    if k>=9: #I-Atom
        global I_list
        I_list=np.append(I_list,[y],axis=0)
        
#Speichern aller Atomkoordinaten in passender Reihenfolge                      
def zusammenfuegen(): 
    global koord
    koord=np.append(koord,Pb_list,axis=0)   
    koord=np.append(koord,C_list,axis=0)
    koord=np.append(koord,N_list,axis=0) 
    koord=np.append(koord,H_list,axis=0)  
    koord=np.append(koord,I_list,axis=0) 

#Berechnene aller Atomkoordinaten fuer die versch. Einheitszellen
#gehe Zellen in z-Richtung durch
for zloop in range(0,zAnz): 
    #gehe Zellen in y-Richtung durch
    for yloop in range (0,yAnz): 
        #gehe Zellen in x-Richtung durch
        for xloop in range (0,xAnz): 
            if mode==1: #Winkeleingabe per Hand
                PHI=float(input('phi in Grad (z-Achse)= '))
                THETA=float(input('theta in Grad(y-Achse)= '))
                PSI=float(input('psi in Grad (x-Achse)= '))
            if mode==2: #zufaellige Winkel
                PHI=round(rand.uniform(0,360),2)
                THETA=round(rand.uniform(0,360),2)
                PSI=round(rand.uniform(0,360),2)
                #spreichere Zufallswinkel in Datei
                fwinkelRand.write('%4d %4d %4d %8.2f %8.2f \
                %8.2f\n'%(xloop+1,yloop+1,zloop+1,PHI,\
                THETA,PSI))
            if mode==3: #Ausrichtung in a-Richtung
                PHI=0
                THETA=0
                PSI=0
            if mode==4: #Winkel aus Datei "Zufallswinkel"
                winkel=fZufWinkel.readline().split()
                PHI=float(winkel[3])
                THETA=float(winkel[4])
                PSI=float(winkel[5])
            for i in range(0,12): 
                z=np.empty([3,])
                #Drehe nur Molekuel nicht Pb und I
                if (i>=1 and i<9): 
                    #Rotieren des Molekuels
                    x=rotation(PHI,THETA,PSI,b[i]) 
                    x=x+0.5 #verschieben des Molekuels 
                    #in die Mitte der einzelnen Zelle
                else: #Koordinaten der Pb/I-Atome
                    x=b[i] 
                #Verschieben der einzelnen Atome in
                #passende Zelle
                z[0]=aG*(x[0]+xloop)
                z[1]=aG*(x[1]+yloop)
                z[2]=aG*(x[2]+zloop)
                speichern(i,z)
            
#Erstelle Liste aller Atomkoord. in richtiger Reihenfolge            
zusammenfuegen()


#Schreibe POSCAR-file
f.write("MAPbI3\n",) #Kommentar
f.write('%4.2f' %(a)) #Faktor
f.write('\n') 
#Gittervektoren
f.write('%12.6f %12.6f %12.6f\n'%(a1[0],a1[1],a1[2]))
f.write('%12.6f %12.6f %12.6f\n'%(a2[0],a2[1],a2[2]))
f.write('%12.6f %12.6f %12.6f\n'%(a3[0],a3[1],a3[2]))
f.write(atoms) #Atomsorten
#Anzal der einzelnen Atome
f.write('%3d %3d %3d %3d %3d\n' %(anzahl[0],anzahl[1],anzahl[2]\
       ,anzahl[3],anzahl[4])) 
f.write("Cartesian\n")
#schreibe Koordinaten der Atome
for i in range(len(koord)):
    f.write('%12.6f %12.6f %12.6f\n'%(koord[i,0],koord[i,1],\
    koord[i,2]))
                
f.close()
fkoord.close()
if mode==2:
    fwinkelRand.close()
if mode==4:
    fZufWinkel.close()
    
    
