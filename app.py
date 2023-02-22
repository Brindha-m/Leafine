# modified from : https://github.com/thepbordin/Obstacle-Detection-for-Blind-people-Deployment/blob/main/app.py

import streamlit as st
import torch
import detect
from PIL import Image
from io import *
import glob
from datetime import datetime
import os
import wget
import time
from streamlit_card import card
from annotated_text import annotated_text
import cv2
from grad_cam import SaveFeatures, getCAM, plotGradCAM



def get_accuracy_str(raw_list): # display detection result as string 

    label_found = []

    for each_label in raw_list:
        label_found.append([each_label[-1], round((each_label[-3]*100),2) ])

    label = []

    for i in range(len(label_found)):
        label.append(label_found[i][0])

    label = list(set(label)) # save a only 1 found label 


    current_label = ""

    count = 0
    
    sum_result = []
    
    for j in range(len(label_found)): # get only 1 labels and most accuracy in result_list
        for k in range(len(label)):
            #print(label_found[j], label[k])
            if (label_found[j][0] == label[k]):
                if current_label != label[k]:
                    current_label = label[k]
                    count += 1
                    if count <= len(label):
                        sum_result.append(label_found[j])#, label[k])
     
    

    if len(sum_result) > 0:
        for each_label in sum_result: # print detection result
               
                ######## Powdery Mildew ###################

            if each_label[0] == "Powdery Mildew":
                label_name = "Powdery Mildew"
                st.success(f'Chance is : "{label_name}" {each_label[1]} % ')

                recommend = "🌱 If a plant displays signs of a disease, remove all the infected parts and destroy them by burning.\n 🌱 Replace the soil in the flowerpot.\n 🌱 Use only settled room-temperature water forwatering.\n 🌱 Adjust the air conditions. If houseplants are infected,keep them spaced."
                treatment = "- After harvesting the crop To destroy the plant debris that used to be diseased by tilling. And crop rotation. \n - Spraying fungicides such as triadimefon, myclobutanil. (myclobutanil) propiconazole (propiconazole) azocystrobin (azoxystrobin)"
                st.image("./replant/Powdery1.png")
                st.image("./replant/PowderyPM.png")
                st.image("./replant/PowderyI.png")
                st.image("./replant/PowderyHM.png")
                st.image("./replant/ProductRecom1.png")

                col1, col2 = st.columns(2)
                col3, col4 = st.columns(2)

  
                with col1:
                       #st.write('Caption for second chart')
                       hasClicked = card(
            title="Powdery Prochloraz",
            text="Midazole fungicide that is widely used in gardening and agriculture",
            image="https://cdn.shopify.com/s/files/1/0722/2059/products/4copy_1800x1800.webp?v=1672229156",
            url="https://www.bighaat.com/products/score-fungicide?variant=12725272936471&currency=INR&utm_medium=product_sync&utm_source=google&utm_content=sag_organic&utm_campaign=sag_organic&utm_source=Google&utm_medium=CPC&utm_campaign=17706716593&utm_adgroup=&utm_term=&creative=&device=c&devicemodel=&matchtype=&feeditemid=&targetid=&network=x&placement=&adposition=&GA_loc_interest_ms=&GA_loc_physical_ms=1007810&gclid=Cj0KCQiAofieBhDXARIsAHTTldpXkjTg0o32bEGopU2HNKUUZVseCAvqWfX6tgApx_MFEWtPNGi8cu4aAjLhEALw_wcB"
        )      
                    
                with col2:
                    hasClicked1 = card(
            title="Patch Pro Fungicide",
            text="Systemic fungicide that contains the active ingredient Propiconazole",
            image="https://smhttp-ssl-60515.nexcesscdn.net/media/catalog/product/cache/1/image/600x600/9df78eab33525d08d6e5fb8d27136e95/p/a/patch_pro_grn_shadow2_1.jpeg",
            url="https://www.solutionsstores.com/patch-pro-fungicide"
        )
                with col3:
                    hasClicked2 = card(
            title="SAAF Fungicide",
            text="Controls Anthracnose, Powdery mildew AND Rust Disease",
            image="https://cdn.shopify.com/s/files/1/0722/2059/products/Untitleddesign_1_adec1180-1362-4f71-a633-1235b3b9e313_800x.jpg?v=1668517899",
            url="https://www.bighaat.com/products/upl-saaf-fungicide?variant=31478554722327&currency=INR&utm_medium=product_sync&utm_source=google&utm_content=sag_organic&utm_campaign=sag_organic&utm_source=Google&utm_medium=CPC&utm_campaign=16667009224&utm_adgroup=&utm_term=&creative=&device=c&devicemodel=&matchtype=&feeditemid=&targetid=&network=x&placement=&adposition=&GA_loc_interest_ms=&GA_loc_physical_ms=1007810&gclid=Cj0KCQiA54KfBhCKARIsAJzSrdoz8vri-PqBXgXRx7JCt1TEZFVXPtt4PRoj_KxcRXxc4xCzhrKmc9saAuhFEALw_wcB"
        )
                with col4:
                    hasClicked3 = card(
            title="Leemark",
            text="Prevent Powdery mildew, Black spot, Downy mildew, Blights & Molds",
            image="https://badikheti-production.s3.ap-south-1.amazonaws.com/products/202301281242101383789267.jpg",
            url="https://www.badikheti.com/organic-pesticide/pdp/leemark-prevent-powdery-mildew-black-spot-downy-mildew-blights-molds-and-other-plant-diseases/269unrka"
        )
             
                ######## LEAF SPOT ###################
                
            elif each_label[0] == "Spot":
                label_name = "Leaf Spot Disease"
                st.success(f'Chance is : "{label_name}" {each_label[1]} % ')

                recommend = "- Prune durian branches to make it airy. \n - When the branches and leaves show few symptoms of the disease, cut and collect and burn. Including collecting fallen leaves and burning them to reduce the accumulation of pathogens. and reduce the outbreak in the following year."
                treatment = "- Spray with anti-fungal agents such as prochloraz, mancozeb, diphenoconazole. (difenoconazole), etc."
                
                st.image("./replant/Spot1.png")
                st.image("./replant/SpotPM.png")
                st.image("./replant/SpotI.png")
                st.image("./replant/SpotCause.png")
                st.image("./replant/ProductRecom1.png")

                col1, col2 = st.columns(2)
                col3, col4 = st.columns(2)

  
                with col1:
                       #st.write('Caption for second chart')
                       hasClicked = card(
            title="GreenDrop",
            text="Enrich the soil and enhance beneficial microbes helps in nitrogen fixation.",
            image="https://plantic.in/image/greendrop.jpg",
            url="https://plantic.in/products/organic-greendrop?source=google&medium=cpc&campaignid=16514436694&adgroupid=&keyword=&matchtype=&device=c&gclid=Cj0KCQiA54KfBhCKARIsAJzSrdphpQ-yJVyyxVQLz3Q8RBR0G8LC3Ai_5os9tlLhbu9gyfZqwwRSU6waAhWTEALw_wcB"
        )      
                    
                with col2:
                    hasClicked1 = card(
            title="Herbal Garden Protection",
            text="Herbal water based Eco-friendly spray made from Aromatic Oils and plant extracts.",
            image="https://cdn.shopify.com/s/files/1/0577/8951/3913/products/HBPA38-1_900x.jpg?v=1639130544",
            url="https://herbalstrategi.com/products/herbal-garden-protection-spray-for-pest-and-fungi-protection-500-ml-wellness-spray-bio-spray-for-faster-plant-growth-500-ml?variant=42162982387950&utm_source=google&utm_medium=cpc&utm_campaign=Google+Shopping&currency=INR&gclid=Cj0KCQiA54KfBhCKARIsAJzSrdqBBniS-xFaHcbHe7VmYWn8BgGhbVHyhWLZrnUNfomdLia5rtfEwe8aAi8AEALw_wcB"
        )
                with col3:
                    hasClicked2 = card(
            title="Trichoderma Viride",
            text="Excellent for suppressing diseases caused by fungal pathogens",
            image="https://cdn.shopify.com/s/files/1/0451/1101/7626/products/PhotoRoom_20210727_100138_900x.png?v=1627362786",
            url="https://seed2plant.in/products/trichoderma-viride?currency=INR&gclid=Cj0KCQiA54KfBhCKARIsAJzSrdogp8XpyWqEyWWHsdC9xtNnNlsqyJE12vULNvKn1KEblURtxwFdfMcaAo5IEALw_wcB"
        )
                with col4:
                    hasClicked3 = card(
            title="NPK Fertilizer",
            text="It controls leaf yellowing, improves green leaves and prevent the Black spots",
            image="https://m.media-amazon.com/images/I/71JcFuVYRFL._SX522_.jpg",
            url="https://www.amazon.in/19-Fertilizer-Garden-Plants-0-25/dp/B0B1MZV3DS/ref=asc_df_B0B1MZV3DS/?tag=googleshopdes-21&linkCode=df0&hvadid=586198977745&hvpos=&hvnetw=g&hvrand=10187752443105046934&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=1007810&hvtargid=pla-1722813852198&th=1"
        )
                    

        
         ########### Blight disease - EARLY / LATE  ############


            elif each_label[0] == "Blight":
                label_name = "Blight"
                st.success(f'Chance is : "{label_name}" {each_label[1]} % ')

                recommend = "- Prune the diseased leaves. Including weed control in the planting area. to reduce the accumulation of pathogens"
                treatment = "- Cut out the diseased branches, burn them. (If it is a large branch, it should be applied with red lime or copper compounds) and then sprayed with carbendashim. (carbendazim) 60% WP rate 10 grams per 20 liters of water or copper oxychloride 85% WP rate 50 grams per 20 liters of water throughout the interior and exterior."
              
                st.image("./replant/Blight1.png")
                st.image("./replant/BlightPM.png")
                st.image("./replant/BlightI.png")
                st.image("./replant/BlightCause.png")
                st.image("./replant/BlightHM.png")
                st.image("./replant/ProductRecom1.png")

                col1, col2 = st.columns(2)
                col3, col4 = st.columns(2)

  
                with col1:
                       #st.write('Caption for second chart')
                       hasClicked = card(
            title="Kavach Fungicide",
            text="Contains Chlorothalonil is a broad-spectrum contact fungicide and is highly effective against Anthracnose, Fruit Rots, Early and Late Blight on various crops.",
            image="https://cdn.shopify.com/s/files/1/0722/2059/products/3_35_800x.webp?v=1672228385",
            url="https://www.bighaat.com/products/kavach-fungicide?variant=31592941977623&currency=INR&utm_medium=product_sync&utm_source=google&utm_content=sag_organic&utm_campaign=sag_organic&utm_source=Google&utm_medium=CPC&utm_campaign=17706716593&utm_adgroup=&utm_term=&creative=&device=c&devicemodel=&matchtype=&feeditemid=&targetid=&network=x&placement=&adposition=&GA_loc_interest_ms=&GA_loc_physical_ms=1007810&gclid=Cj0KCQiA54KfBhCKARIsAJzSrdqiXU6Un3-UNUfDmbDkNTItG7qie77l235Xk5ANEddTTVFiPuUpH9AaAs0XEALw_wcB"
        )      
                    
                with col2:
                    hasClicked1 = card(
            title="MYCICON",
            text="Fungicide For Controlling Fungal Infection Like Powdery, Downy Mildew And Blight",
            image="https://encrypted-tbn1.gstatic.com/shopping?q=tbn:ANd9GcQ1AD4mlIQHnyau2MjSmqOXEKv4GwUuATWcMquOM4GF94nQz7LZcTsj_23qOUzecP7Pf6-rEz2r1BBEZ2kIH0TsX30CNC6CzCT1lV2WFHnGwcmBUfZCW56dog&usqp=CAc",
            url="https://agribegri.com/products/svfvgf.php"
        )
                with col3:
                   hasClicked2 = card(
            title="Green-Drop",
            text="Enrich the soil and enhance beneficial microbes helps in nitrogen fixation.",
            image="https://plantic.in/image/greendrop.jpg",
            url="https://plantic.in/products/organic-greendrop?source=google&medium=cpc&campaignid=16514436694&adgroupid=&keyword=&matchtype=&device=c&gclid=Cj0KCQiA54KfBhCKARIsAJzSrdphpQ-yJVyyxVQLz3Q8RBR0G8LC3Ai_5os9tlLhbu9gyfZqwwRSU6waAhWTEALw_wcB"
        )      
                with col4:
                    hasClicked3 = card(
            title="NPK 19-19-19 Fertilizer",
            text="It controls leaf yellowing, improves green leaves and prevent the Black spots",
            image="https://m.media-amazon.com/images/I/71JcFuVYRFL._SX522_.jpg",
            url="https://www.amazon.in/19-Fertilizer-Garden-Plants-0-25/dp/B0B1MZV3DS/ref=asc_df_B0B1MZV3DS/?tag=googleshopdes-21&linkCode=df0&hvadid=586198977745&hvpos=&hvnetw=g&hvrand=10187752443105046934&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=1007810&hvtargid=pla-1722813852198&th=1"
        )
                    

           
           
            elif each_label[0] == "N_loss":
                label_name = "Nitrogen Deficiency Symptoms"
                st.success(f'Chance is : "{label_name}" {each_label[1]} % ')

                recommend = "None"
                treatment = "- Soil fertilization: Mix NPK fertilizer with the highest N ratio and observe the amount of application according to the symptoms of the leaves. \n - Foliar fertilization: Use chemical fertilizers with high N values ​​or use urea. Swimming, high nitrogen formula Mix and get out of the water fertilizer."
                st.image("./replant/N1.png")
                st.image("./replant/NI.png")
                st.image("./replant/NPM.png")
                st.image("./replant/ProductRecom1.png")

                col1, col2 = st.columns(2)
  
                with col1:
                       #st.write('Caption for second chart')
                       hasClicked = card(
            title="RAW- Nitrogen, Plant Nutrient",
            text="For treating deficiencies, increase plant growth during vegative Stage, plant feeding supplement, for Indoor Outdoor Use",
            image="https://m.media-amazon.com/images/I/61zSF3cO7HL._AC_SX522_.jpg",
            url="https://www.amazon.com/NPK-Industries-717891-RAW-Nitrogen/dp/B00UL3YGWO?th=1"
        )      
                    
                with col2:
                    hasClicked1 = card(
            title="AquaNature Flora-N",
            text="High quality fertilizer for freshwater planted aquariums",
            image="http://www.aquanatureonline.com/wp-content/uploads/2021/08/FLORA-N.jpg",
            url="http://www.aquanatureonline.com/product/aquanature-flora-n-concentrated-nitrogen-supplement-for-planted-aquarium/?attribute_pa_size-upto=250ml"
        )
             
                
            # st.success(f'Chance is : "{label_name}" {each_label[1]} % ')
            
            # if recommend != "None":
            #     st.write(f"Treatment Instructions : {label_name}")
            #     st.write(recommend)
            # st.write(f"Method of treatment : {label_name}")
            # st.write(treatment)
            
           

    else: # No disease found in the picture.
        st.warning("No disease found in the picture !! Please take a New photo")



def imageInput(device):
    image_file = st.file_uploader(label = "Upload image of the leaf here.. ",type=['png','jpg','jpeg'])
    if image_file is not None:

        st.caption("### Detection result.. (Summary of the analysis)")
        

        #img = Image.open(image_file)
        ts = datetime.timestamp(datetime.now())
        imgpath = os.path.join('data/uploads', str(ts) + image_file.name)
        outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
        with open(imgpath, mode="wb") as f:
            f.write(image_file.getbuffer())
            gram_im = cv2.imread(imgpath)
            st.caption("Grad-Cam view of the leaf")
            st.image(gram_im, width=528, channels="RGB")


        # call Model prediction--
        
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/leafine.pt', force_reload=True)
        _ = model.cuda() if device == 'cuda' else model.cpu() # hide cuda_cnn display source : https://stackoverflow.com/questions/41149781/how-to-prevent-f-write-to-output-the-number-of-characters-written
        pred = model(imgpath)
        pred.render()  # render bbox in image
        for im in pred.ims:
            im_base64 = Image.fromarray(im)
            im_base64.save(outputpath)

        pred.save()
        detect_val = (pred.pandas().xyxy[0]).values.tolist()



        # --Display predicton / print result
        img_out = Image.open(outputpath)
        st.image(img_out)
       
        get_accuracy_str(detect_val) # get detection string result

    else:
        st.caption("## Waiting for image.. ")
     
    
    
                

def main(): 
    #Logo image here
    st.sidebar.image("./replant/logoleafine.png")
    st.sidebar.header("Plant Wellbeing Assistant🍃")
 
    #st.sidebar.title('⚙️ Select option')
    activities = ["Detection (analyzed disease)", "About (about)"]
    choice = st.sidebar.selectbox("Click here to know more..",activities)

    #st.sidebar.markdown("https://bit.ly/3uvYQ3R")

    if choice == "Detection (analyzed disease)":
        st.image("./replant/logoleafine.png")
        # Perceive the leaf ailment and sort out some way to treat them
        st.caption('### Recognize & Perceive the leaf illness and figure out how to treat them!')

        col1, col2 = st.columns(2)

        # with col2:
        #     # option = st.sidebar.radio("Select input type.", ['Image', 'Video'])
        #     if torch.cuda.is_available():
        #         deviceoption = st.radio("Select runtime mode :", ['cpu', 'cuda (GPU)'], index=1)
        #     else:
        #         deviceoption = st.radio("Select runtime mode :", ['cpu', 'cuda (GPU)'], index=0)
        #     # -- End of Sidebar
        # with col1:
        #     pages_name = ['Upload own data']
        #     page = st.radio('Select option mode :', pages_name) 
        
        page = 'upload own data'

        # if page == "Upload own data":
        st.subheader('🔽Upload Image 📸')
        t1 = time.perf_counter()
        #deviceoption = st.radio("Select runtime mode :", ['cpu', 'cuda (GPU)'], index=0)
        deviceoption = 'cpu'
        imageInput(deviceoption)
        t2 = time.perf_counter()
        st.success('Time taken to run: {:.2f} sec'.format(t2-t1))

      

    
    elif choice == 'About (about)' :
        st.image("./replant/N1.png")
        # appgrad.main()

if __name__ == '__main__':
    main()

