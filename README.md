# Detecção de Objetos em tempo real com drones

## ATENÇÃO! Este repositório está sendo atualizado com as intruções, porém os códigos já estão disponibilizados na sua íntegra.

---

***Estou atualizando as intruções completas, mas o script está disponível na sua íntegra no arquivo `main.py`.***

### RESUMO 

1. Criei um hotspot a partir do celular e conectei meu notebook nele, mesma rede.
2. Instalei NGINX no notebook para criar um RTMP Server.
3. Fiz streaming da tela do celular para o notebook, usando o app de celular PRISM Live.
4. Executei um script Python para monitorar a porta 1935 e capturar os pacotes RTMP com o OpenCV.
5. Passei os frames na arquitetura deep learning YOLO, treinada no COCO dataset.
