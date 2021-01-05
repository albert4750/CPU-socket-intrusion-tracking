# CPU socket intrusion tracking
 The code is allowed to be uploaded to github by manager of internship. 

 This is a proposed conceptual design for solving the problem of assembly line operator accidentally damaging 
 motherboard's cpu socket during packaging by warning when their fingers get close to the socket. Damages are
 usually inflicted when the operators accidentally closed the socket lid with parts of their finger or glove 
 still in the socket, crushing the pins in the process.

 The socket is found through 3 different ways, each displayed in the three files "detectGlove.py", 
 "detectGloveV2.py", and "detectGloveV3.py". 

 V1 uses template matching.
 V2 uses object tracking.
 V3 uses SIFT.

 V3's performance is the best.
 

