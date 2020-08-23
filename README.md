# face_detecter
> #개요  
    얼굴의 68개 landmark를 찾고, png이미지를 overlay   
> #사용 라이브러리   
    numpy   
    cv2
    dlib       
    
    *dlib 학습모델 데이터
        https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2   
> # 결과   
* ##얼굴 영역 찾기   
 ![사각형](https://user-images.githubusercontent.com/46700771/90974504-b982bf80-e566-11ea-8b86-a2c302f36dcf.PNG)    
* ##얼굴의 68개의 점        
![얼굴 특징점](https://user-images.githubusercontent.com/46700771/90974514-f8187a00-e566-11ea-8e29-c438ba498b4e.PNG)   
* ##얼굴 센터와 왼쪽상단, 오른쪽하단 찾기       
![얼굴의 센터점 추출](https://user-images.githubusercontent.com/46700771/90974543-4fb6e580-e567-11ea-86dc-75b9108d30d7.PNG)    
* ##overlay 결과   
![overlay](https://user-images.githubusercontent.com/46700771/90974692-614cbd00-e568-11ea-8db8-92a969c4051c.PNG)   
* ##최종 결과      
![결과](https://user-images.githubusercontent.com/46700771/90974650-2054a880-e568-11ea-9854-aa256eebffe6.PNG)   