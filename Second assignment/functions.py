import numpy as np
import cv2
from sklearn import metrics
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


def findRotationAngle(x):
    # Θόλωση της εικόνας
    blurred = cv2.GaussianBlur(x, (9, 9), cv2.BORDER_DEFAULT)
    #show both original and blurred imaget
    #cv2.imshow(np.hstack((x, blurred)))

    #Discrete Fourier Transform
    dft = cv2.dft(np.float32(blurred), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    # Calculate Magnitude Spectrum
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))

    
    # Find the maximum frequency from the the magnitude spectrum
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(magnitude_spectrum)

    # Estimation of rotation angle
    angle_estimate = np.degrees(np.arctan2(max_loc[1] - magnitude_spectrum.shape[0]//2, 
                                       max_loc[0] - magnitude_spectrum.shape[1]//2))

    
    # Find the projection of brightness for the original rotated image
    max_var = np.var(cv2.reduce(blurred, 1, cv2.REDUCE_AVG).reshape(-1))
    best_angle = angle_estimate

    # Try different angles around the estimated one ,rotate the image for each one and find its projection of brightness
    for angle in np.arange(angle_estimate - 10, angle_estimate + 10, 0.05):
        rotated = rotateImage(blurred, angle)
        rotated_var = np.var(cv2.reduce(rotated, 1, cv2.REDUCE_AVG).reshape(-1))
    
        # If the variation of the projection of brightness is bigger this angle is better
        if rotated_var >= max_var:
            max_var = rotated_var
            best_angle = angle


    return best_angle

def rotateImage(image, angle):
    #Center of rotation is the center of the image
    image_center = tuple(np.array(image.shape[1::-1]) / 2)# image_center = (x.shape[1] / 2, x.shape[0] / 2)
    # Rotation Transformation
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    # Apply of the rotation transformation
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return result



def getcontour(img):

    #Preprocess the image: To enhance the contours, you may need to apply certain preprocessing steps. 
    
    kernel = np.ones((1,1),np.uint8)
    # Apply thresholding to get one binary  image
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    #is instead of doing the following but with better result because it had double contour like this
    #dilation = cv2.dilate(img,kernel,iterations = 1)
    #kap=dilation-img

    # Opening on  image
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    #thinn the image
    thinned = cv2.ximgproc.thinning(opening)


    #Find contours: Find the contours in the image using the findContours function provided by OpenCV. 
    #This function returns a list of contours detected in the image and their hierarchy.
    contours,hierarchy = cv2.findContours(thinned, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #cv2.imshow(thinned)
    # Create a cell array to store the contours
    c = []
    for contour in contours:
        c.append(contour.reshape(-1, 2))
    return c,hierarchy

def contour_descriptor(c,N):

    #Create complex sequence
    r = [complex(pt[0], pt[1]) for pt in c]

    # Create a new array with the same length
    x_new = np.linspace(0, len(r), N, endpoint=False)

    #Linear interploate to the desired length
    r_new = np.interp(x_new, np.arange(len(r)), r)
    #f = interp1d(np.arange(len(r)), r, kind='linear')(x_new)

    # Calculate DFT
    R = np.fft.fft(r_new)

    #Calculate the absolute values of R excluding the first element
    descriptor = np.abs(R[1:])

    return descriptor

def findDescriptors(contours,hierarchy,N,multiplier):
  #hiearchy of the contours has the following form [Next,Previous,First_Child,Parent]for each contour and each value represents the index of the contour
  #if a contour doesnt have any of this attributes it has value -1
  #checks how many letters exist in this array of contours by counting the contours that don't have child or parent in the hierarchy
  letters = 1
  if len(contours)>1:
    for contour in hierarchy[0]:
      if contour[0]>0 and contour[3]<0:
        letters+=1
  letters_contours=[]
  letter_index=0
  for i in range(letters):
    level=0
    #it skips them when the contours are small consecutive indicating that they are not letters,it doesnt skip dots,commas because they are recognised as solo chars and they will not have next contour
    #the problem is to have them together if not to be one letter,is is for the case of the  wave underline  in text1.png
    #if letters>1 is to ensure dots and commas are not skipped
    if letters>1 and len(contours[letter_index])<8.5 *multiplier:#a threshold based on experiments
      #if there is next contour after it change the letter_index and continue to this one
      if hierarchy[0][letter_index][0]>0:
        letter_index=hierarchy[0][letter_index][0]
      continue
    letter_descs=[]
    #add the descriptor of the first contour to letter_desc list that represents the descriptors of the letter
    letter_descs.append(contour_descriptor(contours[letter_index],N[level]))
    #checks if the contour has a child and iterate over all the consecutive child and adds them to the letter_descs lilst
    if hierarchy[0][letter_index][2]>0:#if it has child
      child_index=letter_index
      while hierarchy[0][child_index][2]>0:
        if level<2:
          level+=1
        child_index=hierarchy[0][child_index][2]
        letter_descs.append(contour_descriptor(contours[child_index],N[level]))
        #if the child has a next contour/case for up and down children of a letter like "g" or 8 irerates to all the next of it and add the 
        #descriptors to the list
        #and if the next child isn't too small indicating is not a part of the letter
        if hierarchy[0][child_index][0]>0  and len(contours[hierarchy[0][child_index][0]])>=8.5 *multiplier:
          next_child_index=child_index
          while hierarchy[0][next_child_index][0]>0:
            if level<2:
              level+=1
            next_child_index=hierarchy[0][child_index][0]
            letter_descs.append(contour_descriptor(contours[next_child_index],N[level]))

      
    #if we have more than one letters and its not the last one and is next contour is small then is the case of i j
    #append this to the letter_descs too and check the next contour then
    if letters>1 and i<letters-1 and len(contours[hierarchy[0][letter_index][0]])<8.5 *multiplier and hierarchy[0][letter_index][2]<0:
      letter_descs.append(contour_descriptor(contours[hierarchy[0][letter_index][0]],N[level+1]))
    letters_contours.append(letter_descs)
    if hierarchy[0][letter_index][0]>0:
      letter_index=hierarchy[0][letter_index][0]#the next index in the hierarchy for the letter
  letters_contours.reverse()
  return letters_contours


def distance(desc1, desc2):
  #returns the distance between to descriptors
  return np.sum((desc1 - desc2) ** 2)

def read_text_file(filename,encoding):
  #read a text file and convert its text to an array list that each element is a list that contains the words of each line
  lines = []
  #convert text to a list of lines texts
  with open(filename, 'r',encoding=encoding) as file:
      for line in file:
          line = line.strip()
          if line:
              lines.append(line)

  # Split each line into words arrays
  words = [line.split() for line in lines]

  # Convert to NumPy array
  array = np.array(words)
  #close the file
  file.close()

  return array

def preprocess_image(image):
    # Preprocess and edit of the image-undo rotation and return with it the vertical projection of brightness
    angle=findRotationAngle(image)
    cor=rotateImage(image, angle)
    blurred = cv2.GaussianBlur(cor, (9, 9), cv2.BORDER_DEFAULT)
    vertical_projection = cv2.reduce(blurred, 1, cv2.REDUCE_AVG)
    return cor,vertical_projection

def segment_lines(vertical_projection):
    #Spots and seperate the lines of the text based on a threshold on the vertical projection of brightness
    lines = []
    line_start = None
    for i, projection in enumerate(vertical_projection):
        if line_start is None and projection < 255:
            line_start = i
        elif line_start is not None and projection >= 255:
            line_end = i
            lines.append((line_start, line_end))
            line_start = None
    if line_start is not None:
        line_end = len(vertical_projection)
        lines.append((line_start, line_end))
    #I do this to remove the black lines in the top and bottom of the image by finding the mean length of the lines 
    meanLineLength=np.mean([(line[1]-line[0]) for line in lines])
    for i,line in enumerate(lines):
      if (line[1]-line[0])< (0.5*meanLineLength):
        del lines[i]
    return lines


def segment_words(line,multiplier):
    # Seperate the line to words based on the horizontal projection of brightness and one threshold
    words = []
    start = None
    end = None
    threshold =248  
    for i, projection in enumerate(line):
        if start is None and projection < threshold:
            start = i
        elif start is not None and projection >= threshold:
            end = i
            words.append((start, end))
            start = None
            end = None
    if start is not None:
        end = len(line)
        words.append((start, end))
    #remove random character that are not words or letters,only needed in the rotated one
    for i,word in enumerate(words):
      if word[1]-word[0]<np.ceil(multiplier+0.05):#4 for 1496x1698 ,2 for 959x720 ,6 for 3551x4416,this threshold came from experimenting for this sizes
        del words[i]

    #connect the letters that should belong in the same word  based on a minimum distance between consecutive words
    lastword=words[len(words)-1]
    for i in range(len(words)-1):
      stop=False
      while stop==False and words[i+1][0]-words[i][1]<4*np.ceil(multiplier):#16 for 1496x1698 ,4 for 959x720,24 for 3551x4416 ,3 for rotated text1 
        words[i]=(words[i][0],words[i+1][1])
        if words[i+1]== lastword:
          stop=True
        del words[i+1]
      if stop==True or words[i+1]== lastword:
        break
    return words

def segment_characters(word,multiplier):
    #Seperation of the words in characters based on the horizontal projection of brightness of the word
    characters = []
    start = None
    end = None
    threshold =245  #threshold based on experience
    for i, projection in enumerate(word):
        if start is None and projection < threshold:
            start = i
        elif start is not None and projection >= threshold:
            end = i
            characters.append((start, end))
            start = None
            end = None
    if start is not None:
        end = len(word)
        characters.append((start, end))
    #removes too small characters that probably they are pixel with wrong values
    for i,char in enumerate(characters):
      if char[1]-char[0]<2*multiplier/3:
        del characters[i]
    return characters


def create_datasets(image,text_file,N,encoding):
  #takes the argument N which is an array with the sizes for the descriptors of the contours for each level 
  #takes as argument the image and its corresponing file and creates 3 dataset for the 3 classes of letters with fetures the descriptors of the letters 
  #and as labels the corresponding char from the text file 
  #encoding is the argument for the encoding of the input text which is about to be read

  #reads the text file and convert its text to an array list that each element is a list that contains the words of each line
  #in order to match each char of the words to its corresponding  representation in the image 
  text=read_text_file(text_file,encoding)
  dataset1 = []
  dataset2 = []
  dataset3 = []
  #Calculate a multiplier coefficient for the thresholds that exist in the functions segment_words,segment_characters and findDescriptors
  #the caclulation of the multiplier came from experiments with values and it works for the size of the images ,text1.png,text2_150dpi_rot.png,text1_v2.png,text1_v3.png
  base=690480 #959x720 
  image_pixels=image.shape[0]*image.shape[1]
  if image_pixels>base:
    multiplier=(np.log10(image_pixels/base)+1)*2.5
  else:
    multiplier=1
  cor,vertical_projection = preprocess_image(image)
  #segment the image in lines and iterate through each line ,iterate to each line of the text array
  lines = segment_lines(vertical_projection)
  for line_index,line in enumerate(lines):
    horizontal_projection=cv2.reduce(cor[line[0]:line[1],:], 0, cv2.REDUCE_AVG).flatten()
    #segment each line to words and iterate through them ,iterate though each word in the text array  
    words=segment_words(horizontal_projection,multiplier)
    for word_index,word in enumerate(words):
      #the corresponing text of the segmented word from the image
      word_text=text[line_index][word_index]
      chars=0
      #segment each word to characters and iterate through them
      characters=segment_characters(horizontal_projection[word[0]:word[1]],multiplier)
      for char_start, char_end in characters:
        #cv2.imshow(cor[line[0]:line[1],word[0]+char_start:word[0]+char_end])
        #for each char find its contours and its descriptors
        #in the case that 2 or more characters were recognised as one(case of underline) in the segment_characters function
        # the functions gecontour and findDescriptor can identify that and and return the descriptors for each letter found
        _,thresh=cv2.threshold(cor[line[0]:line[1],word[0]+char_start:word[0]+char_end], 127, 255, cv2.THRESH_BINARY)
        contours,hierarchy=getcontour(thresh)
        letters_descs=findDescriptors(contours,hierarchy,N,multiplier/2)
        #print(word_text[chars])
        #iterate through each set of letter descriptors that were found from findDescriptors and depending on the set length append it to the 
        #corresponing database with the corresponing char of the text array of the word the letter belongs
        for letter in letters_descs:
          if len(letter)==1:
            dataset1.append((letter,word_text[chars]))
          elif len(letter)==2:
            dataset2.append((letter,word_text[chars]))
          else:
            dataset3.append((letter,word_text[chars]))
          chars+=1
  return {1:dataset1,2:dataset2,3:dataset3} 

def readtext(image,classifier1,classifier2,classifier3,N):
  #takes as an argument an image,the array N for the size of the descriptors and the 3 trained classifiers for each class of the letters
  #it goes through the image in the same way with create datasets but instead of labeling each descriptor with each corresponing letter
  #it predicts each char with the trained classifier from the descriptors and it returns a text array of the predicted text structured in the 
  #same way read_text_file function returns the text


  #Calculate a multiplier coefficient for the thresholds that exist in the functions segment_words,segment_characters and findDescriptors
  #the caclulation of the multiplier came from experiments with values and it works for the size of the images ,text1.png,text2_150dpi_rot.png,text1_v2.png,text1_v3.png

  base=690480 #959x720 
  image_pixels=image.shape[0]*image.shape[1]
  if image_pixels>base:
    multiplier=(np.log10(image_pixels/base)+1)*2.5
  else:
    multiplier=1
  cor,vertical_projection = preprocess_image(image)
  #segment the image in lines and iterate through each line 
  lines = segment_lines(vertical_projection)
  detected_text = []
  for line_index,line in enumerate(lines):
    line_text=[]
    horizontal_projection=cv2.reduce(cor[line[0]:line[1],:], 0, cv2.REDUCE_AVG).flatten()
    #segment each line to words and iterate through them
    words=segment_words(horizontal_projection,multiplier)
    for word_index,word in enumerate(words):
      word_text=""
      chars=0
      #segment each word to characters and iterate through them
      characters=segment_characters(horizontal_projection[word[0]:word[1]],multiplier)
      for char_start, char_end in characters:
        #cv2.imshow(cor[line[0]:line[1],word[0]+char_start:word[0]+char_end])
        #for each char find its contours and its descriptors
        #in the case that 2 or more characters were recognised as one(case of underline) in the segment_characters function
        # the functions gecontour and findDescriptor can identify that and and return the descriptors for each letter found
        _,thresh=cv2.threshold(cor[line[0]:line[1],word[0]+char_start:word[0]+char_end], 127, 255, cv2.THRESH_BINARY)
        contours,hierarchy=getcontour(thresh)
        letters_descs=findDescriptors(contours,hierarchy,N,multiplier/2)
        #print(word_text[chars])
        #iterate through each set of letter descriptors that were found from findDescriptors and depending on the set length use the corresponing
        #classifier to predict which char it is and add the result to a string variable that represents a word
        for letter in letters_descs:
          if len(letter)==1:
            char=classifier1.predict(letter)
            word_text+=char[0]
          elif len(letter)==2:
            #makes the 2 descriptors of size (N[0],N[1]) to one array with size(N[0]+N[1])
            flat=flattenTestSet(letter)
            char=classifier2.predict(flat)
            word_text+=char[0]
          else:
            #makes the 3 descriptors of size (N[0],N[1],N[2]) to one array with size(N[0]+N[1]+N[2])
            flat= flattenTestSet(letter)
            char=classifier3.predict(flat)
            word_text+=char[0]
      #appends the string of the word to a list that represents the corresponing line
      line_text.append(word_text)
    #appends the list of the line to a list that represents all the text
    detected_text.append(line_text)
  return np.array(detected_text)           

def extract_features(dataset):
    #extracts the features and the labels from the dataset and returns them
    X = []
    y = []
    for item in dataset:
        X.append(item[0])
        y.append(item[1])
    return np.array(X), np.array(y)

def flattenTrainSet(X_mult):
  #makes the matrix of feature from size (number_of samples,number_of_contours)where the second dimension has #number_of_contours vector with different size
  #to a matrix of (number_of_samples,combined_descriptors)by concatenating the vector of the contours descriptors
  extra_size=0
  if X_mult.shape[1]==2:
    extra_size=X_mult[0][1].shape[0]
  elif X_mult.shape[1]==3:
    extra_size=X_mult[0][1].shape[0]+X_mult[0][2].shape[0]
  #the shape of the new flatten matrix
  X_2dim=np.ones((X_mult.shape[0],X_mult[0][0].shape[0]+extra_size))
  for index,X in enumerate(X_mult):
    values=[]
    for i in range(X.shape[0]):
      for n in X[i]:
        values.append(n)
    values=np.array(values)
    for j,value in enumerate(values):
      X_2dim[index,j]=value
  return X_2dim

def flattenTestSet(X_mult):
  #makes the 2 descriptors of size (N[0],N[1]) to one array with size(N[0]+N[1])
  #makes the 3 descriptors of size (N[0],N[1],N[2]) to one array with size(N[0]+N[1]+N[2])
  X_2dim=[]
  values=[]
  for i in range(len(X_mult)):#(X_mult.shape[0]):
    for n in X_mult[i]:
      values.append(n)
  values=np.array(values)
  X_2dim.append(values)
  return np.array(X_2dim)  

def train_classifier(X_train, y_train):
    #train a KNN classifier on the given data using the distance function
    classifier = KNeighborsClassifier(n_neighbors=3,metric=distance)
    classifier.fit(X_train, y_train)
    return classifier

def evaluate_classifier(classifier,X_test,y_test):
  #evaluate the given classifier in the test set given the required sets
  y_pred = classifier.predict(X_test)
  print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
  #Classification report
  print("Classification Report:")
  cr = metrics.classification_report(y_test, y_pred)
  print(cr)
  return metrics.accuracy_score(y_test, y_pred),cr

