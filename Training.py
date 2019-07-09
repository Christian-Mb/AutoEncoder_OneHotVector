print("hello word")
a = 10
b = 23
print(b + a)
test = "hello"
for lettre in test:
    print(lettre)
if a <= 10:
    a += 1;
    b += a;

print("la valeur de a est", a, "la valeur de b est ", b)

List = []
print("Intial blank List: ")
print(List)

# Creating a List with
# the use of a String
List = ['GeeksForGeeks']
print("\nList with the use of String: ")
print(List)

# Creating a List with
# the use of multiple values
List = ["Geeks", "For", "Geeks"]
print("\nList containing multiple values: ")
print(List[0])
print(List[2])

# Creating a Multi-Dimensional List
# (By Nesting a list inside a List)
List = [['Geeks', 'For'], ['Geeks']]
print("\nMulti-Dimensional List: ")
print(List)

# Creating a List with
# the use of Numbers
# (Having duplicate values)
List = [1, 2, 4, 4, 3, 3, 3, 6, 5]
print("\nList with the use of Numbers: ")
print(List)

# Creating a List with
# mixed type of values
# (Having numbers and strings)
List = [1, 2, 'Geeks', 4, 'For', 6, 'Geeks']
print("\nList with the use of Mixed Values: ")
print(List)
List.append("je suis dans la joie")

for value in List:
    print(value)
while 1:
    try:
        nom = str(input("Entrez votre nom : "))
        prenom = str(input("Entrez votre prenom : "))
        age = int(input("Entrez votre âge : "))
        break
    except ValueError:
        print("Veuillez recommencer")

print("Je m'appelle : ", nom, " ", prenom, " et je suis agé de : ", age, " ans")


class Agent :



    def __init__(nom, prenom, age, sexe):
        nom = nom
        prenom=prenom
        age = age
        sexe= sexe