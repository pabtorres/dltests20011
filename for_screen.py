lista = ["hola", "probando", "ando"]
textfile = open("test_file22.txt", "w")
for element in lista:
    textfile.write(str(element) + "\n")
textfile.close()
