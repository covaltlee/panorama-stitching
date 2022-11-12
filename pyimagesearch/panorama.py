# import the necessary packages
import numpy as np
import numpy.linalg as lin
import imutils
import cv2
import time

class Stitcher:
	def __init__(self):
		# determine if we are using OpenCV v3.X
		self.isv3 = imutils.is_cv3()
		
	def stitch(self, N, img, Center, ratio=0.75, reprojThresh=4.0, showMatches=False):
                # unpack the images, then detect keypoints and extract
		# local invariant descriptors from them
		features = [None]*N
		kps = [None]*N
		for i in range(0,N):
		        (kps[i], features[i]) = self.detectAndDescribe(img[i])
		
		# match features between the two images
		M = [None]*(N-1)
		matches = [None]*(N-1)
		H = [None]*(N-1)
		status =[None]*(N-1)
		for i in range(0,N-1):
		    M[i] = self.matchKeypoints(kps[i+1], kps[i],features[i+1], features[i], ratio, reprojThresh)
		    # if the match is None, then there aren't enough matched
		    # keypoints to create a panorama
		    if M[i] is None:
		        continue
		    # otherwise, apply a perspective warp to stitch the images
		    # together
		    
		    (matches[i], H[i], status[i]) = M[i]
	            
		# inverse Matrix of H
		invH = [None]*(N-1)
		for i in range(0,N-1):
		    invH = np.linalg.inv(H)
            
		# Matrix from Center
		HC = [None]*N
		HC[Center] = np.eye(3)
        
		# Matrix Under Center
		TempH = np.eye(3)
		for i in range(Center-1,-1,-1):
		    TempH = np.dot(TempH,invH[i])
		    HC[i] = TempH
	    
		# Matrix Over Center
		TempH = np.eye(3)
		for i in range(Center+1,N):
		    TempH = np.dot(H[i-1],TempH)
		    HC[i] = TempH
		    
		# Parallel Transformation
		dis = [0, 0]
		for i in range(0,Center):
		    dis[0] = dis[0] + img[i].shape[0]
		    dis[1] = dis[1] + img[i].shape[1]
		    
		PT = np.array([
			[1, 0, dis[0]],
			[0, 1, dis[1]],
			[0, 0, 1]
		])
	
		# Parallel Transformed HC
		PTHC = [None]*N
		for i in range(0,N):
			PTHC[i] = np.dot(PT,HC[i])
			
		# inverse PTHC
		invPTHC = [None]*N
		for i in range(0,N):
			invPTHC[i] = lin.inv(PTHC[i])
			
		# Warped
		imgSize = [0,0]
		for i in range(0,N):
			imgSize[0] = imgSize[0] + img[i].shape[0]
			imgSize[1] = imgSize[1] + img[i].shape[1]
		Warped = [None]*N
		for i in range(0,N):
			Warped[i] = cv2.warpPerspective(img[i], PTHC[i], (imgSize[1], imgSize[0]))
			cv2.imshow("Warped"+str(i), Warped[i])
		'''
		# No Weight
		NoWeightImg = Warped[0]
		for i in range(0,N-1):
			NoWeightImg = self.blend_non_transparent(NoWeightImg,Warped[i+1])
		cv2.imshow("R", NoWeight)
		'''

		# Weight
		
		# shape
		shape = [None]*N
		for i in range(0,N):
			shape[i] = img[i].shape

		#Re,W,T
		# three color RGB
		WeightedImg = np.zeros((imgSize[0], imgSize[1], 3)) 
		imageW = np.zeros((N, imgSize[0], imgSize[1]))

		# Calculate Weight
		Weight = [None]*N 
		for i in range(0,N):
			Weight[i] = np.zeros((shape[i][0],shape[i][1],3))
			for P0 in range(shape[i][0]):
				for P1 in range(shape[i][1]):
					for c in range(0,3):
                                                # Weight = min(x)*min(y) in Rectangle
						Weight[i][P0,P1,c] = min(P0, shape[i][0]-P0)*min(P1,shape[i][1]-P1)/(shape[i][0]/2)/(shape[i][1]/2)
		# Transformed Weight				
		TransW = [None]*N
		for i in range(0,N):
			TransW[i] = cv2.warpPerspective(Weight[i], PTHC[i], (imgSize[1], imgSize[0]))

		# Sum Transformed Weight
		TransWSum = [0]
		for i in range(0,N):
			TransWSum = TransWSum + TransW[i]

		# divide to Normalize
		NormTransW = np.divide(TransW, TransWSum, out=None, where=TransWSum!=0)
		#print("---2. %s seconds ---" %(time.time() - start_time))

		for i in range(0,N):
			WeightedImg = WeightedImg + (NormTransW[i]*Warped[i])/255
						
		# print("---3. %s seconds ---" %(time.time() - start_time))
		# for i in range(0,N):
			# cv2.imshow("W"+str(i), imageW[i])
			#cv2.imshow("T"+str(i), imageT[i])
		cv2.imshow("WeightedImg",WeightedImg)
		
		return WeightedImg

		'''                
		cv2.imshow("newA", newimageA)
		cv2.imshow("W", imageW)
		# check to see if the keypoint matches should be visualized
		if showMatches:
			vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
				status)
        
			# return a tuple of the stitched image and the
			# visualization
			return (result, vis)
		'''
		# return the stitched image
		return result

	def detectAndDescribe(self, image):
		# convert the image to grayscale
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# check to see if we are using OpenCV 3.X
		if self.isv3:
			# detect and extract features from the image
			descriptor = cv2.AKAZE_create()
			(kps, features) = descriptor.detectAndCompute(image, None)

		# otherwise, we are using OpenCV 2.4.X
		else:
			# detect keypoints in the image
			# detector = cv2.FeatureDetector_create("SIFT")
			kps = detector.detect(gray)

			# extract features from the image
			# extractor = cv2.DescriptorExtractor_create("SIFT")
			(kps, features) = extractor.compute(gray, kps)

		# convert the keypoints from KeyPoint objects to NumPy
		# arrays
		kps = np.float32([kp.pt for kp in kps])

		# return a tuple of keypoints and features
		return (kps, features)

	def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
		ratio, reprojThresh):
		# compute the raw matches and initialize the list of actual
		# matches
		matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming(2)")
		rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
		matches = []

		# loop over the raw matches
		for m in rawMatches:
			# ensure the distance is within a certain ratio of each
			# other (i.e. Lowe's ratio test)
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))

		# computing a homography requires at least 4 matches
		if len(matches) > 4:
			# construct the two sets of points
			ptsA = np.float32([kpsA[i] for (_, i) in matches])
			ptsB = np.float32([kpsB[i] for (i, _) in matches])

			# compute the homography between the two sets of points
			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
				reprojThresh)

			# return the matches along with the homograpy matrix
			# and status of each matched point
			return (matches, H, status)

		# otherwise, no homograpy could be computed
		return None

	def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
		# initialize the output visualization image
		(hA, wA) = imageA.shape[:2]
		(hB, wB) = imageB.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
		vis[0:hA, 0:wA] = imageA
		vis[0:hB, wA:] = imageB

		# loop over the matches
		for ((trainIdx, queryIdx), s) in zip(matches, status):
			# only process the match if the keypoint was successfully
			# matched
			if s == 1:
				# draw the match
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

		# return the visualization
		return vis
	def blend_non_transparent(self, face_img, overlay_img):
		# Let's find a mask covering all the non-black (foreground) pixels
		# NB: We need to do this on grayscale version of the image
		gray_overlay = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
		overlay_mask = cv2.threshold(gray_overlay, 1, 255, cv2.THRESH_BINARY)[1]
		# Let's shrink and blur it a little to make the transitions smoother...
		overlay_mask = cv2.erode(overlay_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
		overlay_mask = cv2.blur(overlay_mask, (3, 3))
		# And the inverse mask, that covers all the black (background) pixels
		background_mask = 255 - overlay_mask
		# Turn the masks into three channel, so we can use them as weights
		overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
		background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)
		# Create a masked out face image, and masked out overlay
		# We convert the images to floating point in range 0.0 - 1.0
		face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
		overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))
		# And finally just add them together, and rescale it back to an 8bit integer image
		return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))
