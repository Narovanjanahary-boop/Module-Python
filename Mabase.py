import string

# Valider nombre entier
def valid_int(nombre):
    etat = True
    while etat:
        try:
            nombre = int(nombre)
            return nombre
        except:
            nombre = input("Les lettres et les caractères spéciaux ne sont pas autorisés.\nveuillez réessayer : ")
            
# Valider nombre réel
def valid_float(nombre):
    etat = True
    while etat:
        try:
            nombre = float(nombre)
            return nombre
        except:
            nombre = input("Les lettres et les caractères spéciaux ne sont pas autorisés.\nveuillez réessayer : ")
            
    
            
        


# verifier symbole
def verifier_symbole(nom):
    etat_symb = True
    while etat_symb:
        for i in string.punctuation:
            if i in nom:
                nom = input("Le nom ne doit pas contenir un symbole.\nVeuillez réessayer : ")
                etat_nom = False
                break
            else:
                etat_nom = True
        
        if etat_nom == True:
            etat_symb == False
            return nom
    

# verifier_nombre
def verifier_nombre(nom):
    etat_symb = True
    etat_nb = True
    while etat_nb:
        for i in string.digits:
            if i in nom:
                nom = input("Le nom ne doit pas contenir un nombre.\nVeuillez réessayer : ")
                etat_nom = False
                break
            else:
                etat_nom = True
        
        if etat_nom == True:
            etat_nb == False
            return nom
            
def verifier_continuite():
    print("Vous voulez continuer ?")
    print("1 -Oui\n2-Non")
    choix = input("Entrer 1 ou 2 : ")
    choix = verifier_nombre(choix)
    etat = True
    while etat:
        if choix == 2:
            quite = input("Appuyer sur n'importe quelle touche pour quitter ")
            quit()
        elif choix == 1:
            etat = False
            continuite = True
            return continuite

        else:
            choix = input("Entrer un choix valide : ")
            choix = verifier_nombre(choix)
            
            
def afficher_mes_infos():
    print("Bienvenue sur mon programme")
    print("Je m'appelle NAROVANJANAHARY Solofoniaina Ferdinand\nL1 en informatique, parcours:Developpement d'Application Internetet Intranet(DA2I) à l'EMIT Fianarantsoa.")
    print("Contact : \n  Whatsapp: +261332605103 /+261383167780\nE-mail:rovaferdinand844@gmail.com")
    print("GpA , 005I26")
    
        
            