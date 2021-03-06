# MTI_Detection

Moving Target Indicator

The goal of this algorithm is to detect moving targets from a moving platform. We repurpose the Reed-Xiaoli (RX) anomaly detection algorithm to identify moving objects within a stack of frames. It works as follows:

1. Given N number of frames & GPS/IMU data we register each frame to the middle frame. 
		<br> a. TODO: Currently the implementation uses feature based registration
		<br> b. To register frames using GPS/IMU data we register frame X to Y using the Direction Cosine Matrices by going from Frame X -> Frame X Local Level -> ECEF-> Frame Y Local Level -> Frame Y. 
2. Process the stack of registered image frames using the RX algorithm.
		<br> a. See [here for original RX implementation](https://www.umbc.edu/rssipl/pdf/TGRS/tgrs.anomaly/40tgrs06-chang-proof.pdf) and [my paper here for a short description](https://ieeexplore.ieee.org/document/9506700) of how the RX algorithm works <br>
		     b. The algorithm can be found under ARX.py as the function arx.
3. We identify moving objects as anomalies found by the response of the RX algorithm. This is done by:
		<br> a. Compute the cumulative probability distribution of RX score values
		<br> b. Set the confidence coefficient value to 0.995
		<br> c. Select the first RX score with cumulative probability distribution value greater than the confidence coefficient as the threshold.<br>
4. Find the optical between the stack of registered image frames using GPS/IMU data
		<br> a. TODO: Currently calculated using Lucas Kanade or Farneback algorithm <br> 
5. Threshold RX response further using magnitude of optical flow vectors
6. Find bounding boxes around remaining points using connected components
7. Run NMS to remove overlapping boxes
8. TODO: Apply image tracker to continue identifying objects that may have stopped moving or reduced speed significantly.

There are small details omitted from the explanation for brevity; however, all of this is done under the function *mtiChain* in mti_end2end.py
