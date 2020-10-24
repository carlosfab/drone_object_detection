# Detecção de Objetos em tempo real com drones

## ATENÇÃO! Este repositório está sendo atualizado com as intruções, porém os códigos já estão disponibilizados na sua íntegra.

---

### RESUMO 

1. Criei um hotspot a partir do celular e conectei meu notebook nele, mesma rede.
2. Instalei NGINX no notebook para criar um RTMP Server.
3. Fiz streaming da tela do celular para o notebook, usando o app de celular PRISM Live.
4. Executei um script Python para monitorar a porta 1935 e capturar os pacotes RTMP com o OpenCV.
5. Passei os frames na arquitetura deep learning YOLO, treinada no COCO dataset.


### INSTRUÇÕES (EM ATUALIZAÇÃO)

##### 1. Baixar o modelo YOLO-COCO

Devido à restrição de tamanhos de arquivos no Github, não foi possível disponibilizar o modelo YOLO treinado no dataset COCO.

Para esse script funcionar, [baixe este arquivo zip aqui](https://www.dropbox.com/s/ghe0ksnom26skah/yolo-coco.zip?dl=0) e coloque todo o conteúdo extraído (3 arquivos no total) dentro do diretório `coco-yolo`.
