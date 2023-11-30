# Imports
import cv2
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import sys, os
from skimage.util import random_noise



# VARIAVEIS GLOBAIS

image = cv2.cvtColor(cv2.imread("./img/palmeiras.jpg"), cv2.COLOR_BGR2RGB) # Pré define uma imagem e converte de BGR para RGB
imageName = 'palmeiras.jpg' # Pega o nome da imagem pré definida
imagesDir = "./img" # Define o diretorio das imagens
images = [] # Inicializa o array de imagens
updateNames = []
images.append(image) # Adiciona a imagem pré definida ao array de imagens
updateNames.append("Original") # Adiciona o original no array do rastreio de alterações


def exibir_menu():
    print("\n\t-= Escolha seu processamento =- \n")
    print(f'1 - Contraste e Brilho da Imagem')
    print(f'2 - Segmentação de Imagem')
    print(f'3 - Restauração de Imagem')
    print(f'4 - Adicionar Ruído na Imagem')
    print(f'5 - Gerar Imagem a partir das Bordas')
    print(f'6 - Mostrar Fluxo de Alterações')
    print(f'7 - Selecionar Imagem')
    print(f'8 - Sair')
    print(f"\n\tImagem selecionada: \'{imageName}\'")
    
def exibir_menu_imagens():
    global imagesDir
    global image
    global images
    global updateNames
    global imageName
    
    # LIMPA O CONSOLE
    os.system('cls')
    
    # BUSCA TODAS AS IMAGENS DO DIRETÓRIO
    files = os.listdir(imagesDir)
    
    # Limpa o array de Imagens para não comprometer o fluxo de processamento posteriormente
    images = []
    updateNames = []
    
    print("\n\t-= Escolha sua imagem =- \n")
    
    # EXIBE TODAS AS IMAGENS DISPONIVEIS PARA O USUÁRIO SELECIONAR
    for index, file in enumerate(files):
        print(f"{index+1} - {file}")
    
    imageOption = int(input("\nSelecione uma imagem >> "))
    
    # GUARDA O NOME DA IMAGEM EM UMA VARIAVEL GLOBAL PARA O USUARIO SABER QUAL IMAGEM ELE SELECIONOU
    imageName = str(files[imageOption-1])
    
    # GUARDA A IMAGEM EM UMA VARIAVEL GLOBAL PARA PODER REALIZAR OS PROCESSAMENTOS
    image = cv2.cvtColor(cv2.imread(imagesDir+"/"+files[imageOption-1]), cv2.COLOR_BGR2RGB)
    
    # GUARDA A IMAGEM ORIGINAL NO ARRAY DE IMAGENS PARA EXIBIR NO FLUXO POSTERIORMENTE
    images.append(image)
    updateNames.append("Original")

def contrast_adjustment():
    global image
    
    # new_image = np.zeros(image.shape, image.dtype)
    
    alpha = float(input("\n\n\tDigite quanto de contraste você deseja (1.0/3.0)>> "))
    beta = int(input("\n\n\tDigite quanto de brilho você deseja (0/100)>> "))
    
    
    # Realiza a operação new_image(i,j) = alpha*image(i,j) + beta
    # Ao invés desse looping, poderia ter usado simplesmente o comando abaixo:
    # new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    # Mas nós queriamos mostrar como acessar os pixels 
    # Como cada pixel tem 3 valores, sendo eles o R G B, precisamos percorrer cada um separadamente
    
    # for y in range(image.shape[0]):
    #     for x in range(image.shape[1]):
    #         for c in range(image.shape[2]):
    #             new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
    
    # Observe que o looping acima é bem mais demorado que o comando abaixo da biblioteca openCv
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    # adjusted = new_image

    antes_e_depois(image, adjusted)
    
    image = adjusted
    
    images.append(image)
    updateNames.append("Ajuste de Contraste e Brilho")
    
def image_segmentation():
    global image
    
    # CONVERTE A IMAGEM PARA UMA ESCALA DE CINZA
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # DEFINE OS LIMITES DA IMAGEM
    _ , tresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV   ) # type: ignore
    
    # DEFINE A ALTURA E LARGURA DA IMAGEM
    h, w, *_ = image.shape
    
    # Pega os contornos da imagem a partir dos limites threshold
    contours , hierarchy = cv2.findContours(tresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Pega a maior área da imagem com o seu contorno
    cnt = sorted(contours, key=cv2.contourArea)[-1]

    # Cria uma mascara para aplicar na imagem, baseado na altura e largura da imagem
    mask = np.zeros((h, w), dtype="uint8" )

    # Gera a mascara final da imagem, utilizando a mascara baseado nas proporções da imagem e os contornos, levando em consideração as cores da imagem
    maskedFinal = cv2.drawContours(mask, [cnt] , -1 , (255 , 255 , 255), -1)

    # Gera a imagem final, utilizando ela como referencia e ela como destinataria, aplicando a mascara gerada anteriormente
    finalImage = cv2.bitwise_and(image, image, mask=maskedFinal)

    antes_e_depois(image, finalImage)
    
    # Adiciona a imagem final ao array images, para visualizar posteriormente no fluxo
    images.append(finalImage)
    updateNames.append("Segmentação de Imagem")
    
    # Seleciona a imagem resultante como a imagem atual, para realizar os processamentos nela
    image = finalImage 

def image_denoisy():
    global image

    # Performa a remoção do "noise", isto é, ruído da imagem, utilizando o algoritmo Non-local Means Denoising http://www.ipol.im/pub/algo/bcm_non_local_means_denoising/ com algumas otimizações computacionais.
    # O ruído esperado é do tipo "gausian white noise", aquela mais comum que parece os efeitos de filme antigo.
    dst = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
    
    antes_e_depois(image, dst)
    
    image = dst
    
    
    images.append(image)
    updateNames.append("Restauração de Imagem")

def add_image_noise():
    global image
    
    # ADICIONA O EFEITO GAUSSIAN NOISE, QUE REMETE AO RUÍDO DE UMA IMAGEM E CONVERTE PARA UM NUMPY ARRAY NOVAMENTE O RETORNO
    noisy_image = random_noise(image, mode="gaussian") 
    noisy_image = np.array(255*noisy_image, dtype = 'uint8') 
    
    antes_e_depois(image, noisy_image)
        
    image = noisy_image             
    images.append(image)
    updateNames.append("Adição de Ruído")
    
def images_show():
    fig = plt.figure(figsize=(10, 7)) 
    
    # FAZ UMA VERIFICAO PARA IR ADICIONANDO COLUNAS ATÉ UM MAXIMO DE 3, DE ACORDO COM A QUANTIDADE DE IMAGENS NO ARRAY IMAGES
    rows = 0
    if(len(images) == 1):
        columns = 1
    elif(len(images) == 2):
        columns = 2
    else:
        columns = 3
    
    for i in range(0, len(images), 3):
        rows += 1
              
    
    # FAZ A LEITURA DE TODAS AS IMAGENS CONTIDAS NO ARRAY IMAGES, QUE SAO OS RESULTADOS DOS PROCESSAMENTOS DA IMAGEM ORIGINAL
    
    for index, img in enumerate(images):
        if index == 0:
            fig.add_subplot(rows, columns, 1)
            plt.imshow(img) 
            plt.axis('off') 
            plt.title("Imagem Original")            
        else:    
            fig.add_subplot(rows, columns, (index + 1))
            plt.imshow(img) 
            plt.axis('off') 
            plt.title("Imagem " + str(index + 1) + f" - {updateNames[index]}")            
            
    fig.suptitle('Fluxo de Alterações', fontsize=30)
    
    plt.show()    

def definir_bordas():
    global image
    
    # Utilizamos a função Canny() para detectar edges
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2BGR),100,200)

    # COLOR = [102,102,153]
    COLOR = [0,0,255]

    scale_percent = 40 # percentagem do tamanho original
    width = int(edges.shape[1] * scale_percent / 100)
    height = int(edges.shape[0] * scale_percent / 100)
    dim = (width, height)

    # redimensiona a imagem
    resized = cv2.resize(edges, dim, interpolation=cv2.INTER_AREA)

    # adiciona borda a imagem
    image_edges = cv2.copyMakeBorder(resized, 10, 10, 10, 10,cv2.BORDER_CONSTANT,value=COLOR) 
    
    antes_e_depois(image, image_edges)
    
    images.append(image_edges)
    updateNames.append("Imagem a partir das Bordas")
    image = image_edges

def antes_e_depois(imagemAntes, imagemDepois):
    # Cria uma figura
    
    fig = plt.figure(figsize=(10, 7)) 
    
    # Define a quantidade de linhas e colunas
    rows = 1
    columns = 2
    
    # Adiciona um subplot na figura
    fig.add_subplot(rows, columns, 1) 
    
    # Adiciona a imagem original na figura 
    plt.imshow(imagemAntes) 
    plt.axis('off') 
    plt.title("Antes") 
    
    # Adiciona um subplot na figura
    fig.add_subplot(rows, columns, 2) 
    
    # Adiciona o resultado do processamento da imagem na figura
    plt.imshow(imagemDepois) 
    plt.axis('off') 
    plt.title("Depois")
    
    plt.show()
        
def main():
    global image
    global imageName
    
    # LIMPA O CONSOLE DO SISTEMA
    os.system('cls')
    # EXIBE O MENU DE SELEÇÃO DE IMAGENS
    exibir_menu_imagens()
    
    while True:
        # LIMPA O CONSOLE DO SISTEMA
        os.system('cls')
        # EXIBE O MENU DE PROCESSAMENTOS
        exibir_menu()
        menuOption = int(input("\n\nQual o processo que você deseja executar >> "))
        
        if menuOption == 1:
            contrast_adjustment()            
        elif menuOption == 2:
            image_segmentation()
        elif menuOption == 3:
            image_denoisy()
        elif menuOption == 4:
            add_image_noise()
        elif menuOption == 5:
            definir_bordas()          
        elif menuOption == 6:
            images_show()    
        elif menuOption == 7:
            exibir_menu_imagens()
        elif menuOption == 8:
            os.system('cls')
            sys.exit()            
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()

