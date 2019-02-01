package watershed;


//for ImageJ functionality
import net.imagej.ImageJ;
import ij.gui.GenericDialog;

//ImageJ object classes

import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.Roi;
import ij.gui.Overlay;
import ij.measure.Calibration;
import ij.process.ImageProcessor;
import ij.process.FloatProcessor;
import ij.process.ImageConverter;
import ij.plugin.PlugIn;
import ij.IJ;
import ij.process.ByteProcessor;
import ij.gui.PolygonRoi;
import ij.gui.GenericDialog;
import ij.io.*;
//External JAR features
import sc.fiji.skeletonize3D.*;
//ImageJ-OpenCV conversion objects
import ijopencv.ij.ImagePlusMatConverter;
import ijopencv.opencv.MatImagePlusConverter;
//OpenCV Java objects
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.opencv_core.CvType;
import org.bytedeco.javacpp.opencv_core.KeyPointVector;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_features2d;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.Point2f;
import org.bytedeco.javacpp.opencv_core.KeyPoint;
import org.bytedeco.javacpp.opencv_imgproc;
import org.opencv.imgproc.Imgproc;
import org.bytedeco.javacpp.opencv_highgui;
import org.bytedeco.javacpp.opencv_imgcodecs;
//Java objects
import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
//more java objects
import java.util.Vector;
import java.util.ArrayList;
import java.util.List;
import java.awt.Component;
import java.awt.Image;
import java.lang.Math;
//tracer files
import watershed.AStar;
import watershed.Node;
/**
*
* @author Mark
*/
@Plugin(type = Command.class, headless = true, menuPath = "Plugins>IJ-OpenCV-plugins>watershed")
public class watershedTransform implements Command {
   
  @Parameter
  private ImagePlus imp;
    
  @Override
  public void run() {	  
	  //convert images to openCV format
	  ImagePlusMatConverter ic = new ImagePlusMatConverter();
	  MatImagePlusConverter mip = new MatImagePlusConverter();	  
	  //Soma detection outputted as a keypoint vector with positions of soma 
	  //centers
	  opencv_core.Mat imageCV = ic.convert(imp,Mat.class);
	  ImagePlus icCopy = imp.duplicate();
	  opencv_core.Mat imageCopy = ic.convert(icCopy,Mat.class);
	  KeyPointVector somas = findSomas(imageCopy);
	  
	  //create dialog box to input values
	  int Tol = 1000;
	  java.lang.String Prompt = "Enter Tolerance: ";
	  IJ.showStatus("Plugin Message Test started.");
	  
	  @SuppressWarnings("resource")
	  KeyPoint p = new KeyPoint();
	  @SuppressWarnings("resource")
	  Point2f p2f = new Point2f();
	  int iKey;
	  int jKey; 
	  int rad;
	  
	  Vector<Integer> iPosSoma = new Vector<Integer>();
      Vector<Integer> jPosSoma = new Vector<Integer>();
      Vector<Integer> radSoma = new Vector<Integer>();
	  for (int l = 0; l < somas.size(); l++) {  
		  p = somas.get(l);
		  p2f = p.pt();
		  rad = (int) p.size();
		  jKey = (int) p2f.y();
		  iKey = (int) p2f.x();
		  
		  iPosSoma.add(iKey);
		  jPosSoma.add(jKey);
		  radSoma.add(rad);	  
	      }	  			  
     //Canny edge detection and thresholding
      opencv_imgproc.Canny(imageCV, imageCV, 60, 210);
      opencv_imgproc.threshold(imageCV, imageCV, 0, 255, opencv_imgproc.THRESH_OTSU);      
      final Size ksize = new Size(2, 2);       
      Mat kernel = opencv_imgproc.getStructuringElement(Imgproc.MORPH_RECT, ksize);
      //dilate image to fill in edges
      opencv_imgproc.dilate(imageCV, imageCV, kernel, new Point(-1,-1), 2, opencv_core.BORDER_CONSTANT, opencv_core.Scalar.BLACK); 
      opencv_imgproc.distanceTransform(imageCV, imageCV, opencv_imgproc.DIST_L2, 5);           
      //now threshold again and turn into binary image (pixel values 255 or 0)
      opencv_imgproc.threshold(imageCV, imageCV, 0.7 * 1, 255, opencv_imgproc.CV_THRESH_BINARY);
      imageCV.convertTo(imageCV, opencv_core.CV_8U, 1, 0);     
      ImagePlus temp_image= mip.convert(imageCV,ImagePlus.class);     
      ImagePlus tp = temp_image.duplicate();
      ImageProcessor fillSomas =  tp.getProcessor();            
  //fill somas with bleached voxels for sucesfull tracing      
      for (int i = 0; i < iPosSoma.size(); i++) {
	  int ii = iPosSoma.get(i);
	  int jj = jPosSoma.get(i);		  
	  	for (int j = 0; j < (int) (radSoma.get(i)/2)+2; j++) {
		   for (int k = 0; k < (int) (radSoma.get(i)/2)+2; k++) {
			  fillSomas.putPixel(ii+j, jj+k, 255);
			  fillSomas.putPixel(ii+j, jj-k, 255);
			  fillSomas.putPixel(ii-j, jj+k, 255);
			  fillSomas.putPixel(ii-j, jj-k, 255);			  
		      }
	      }	  
      }  
  //somas are now filled in thresholded image...ready to trace
  ImagePlus imageToTrace= new ImagePlus(" ", fillSomas); 
  imageToTrace.setTitle("tracing image");
  imageToTrace.show();      
      //get skeleton (possible for 3d images as well)
      Skeletonize3D_ imSKEL = new Skeletonize3D_();
      imSKEL.setup(" ", temp_image);      
      ImageProcessor temp_proc = temp_image.getProcessor();      
      imSKEL.run(temp_proc);       
      ImagePlus outSkel = new ImagePlus("outIM",temp_proc);      
      outSkel.show();      
     // ImagePlus skelDuplicate = outSkel.duplicate();
      ImagePlus skelDupN = outSkel.duplicate();
      ImageConverter conv = new ImageConverter(skelDupN);
      conv.convertToRGB();
      //find endpoints and push subsequent ROI's into image (for visualisation)
      Vector<Integer> iPos = new Vector<Integer>();
      Vector<Integer> jPos = new Vector<Integer>();
      //determine endpoints for inputs to trace search algorithm
      Vector<Vector<Integer>> endpointPositions = findEndpoints(skelDupN);
      iPos = endpointPositions.get(0);
      jPos = endpointPositions.get(1);     
      //time for tracing!!!!!      
      TraceNeuron(imageToTrace, icCopy, iPosSoma, jPosSoma, iPos, jPos,Tol);     
  }
  
  
  
  

 void TraceNeuron(ImagePlus Timage, ImagePlus imToTraceOver, Vector<Integer> iSomaPos, 
		 Vector<Integer> jSomaPos, Vector<Integer> PosI, Vector<Integer> PosJ, int Tolerance) {
	  
	  ImagePlusMatConverter icc = new ImagePlusMatConverter();
	  MatImagePlusConverter mipp = new MatImagePlusConverter();
      //vectors of black pixels used for blocking array	  
      Vector<Integer> blackSquaresi = new Vector<Integer>();
      Vector<Integer> blackSquaresj = new Vector<Integer>();
      int[] bp = new int[4];
      for(int q = 0; q < Timage.getHeight(); q++) {
    	  for(int ll = 0; ll < Timage.getWidth(); ll++) {
    		  bp =  Timage.getPixel(q, ll);
        	  if(bp[0] < 50) {
        		  blackSquaresi.add(q);
        		  blackSquaresj.add(ll);       		  
        	  }       	  
          }    	  
      }
         
      int[][] blocksArray = new int[blackSquaresi.size()][2];   
      for (int rr = 0; rr < blackSquaresi.size(); rr++) {
    	  blocksArray[rr][0] = blackSquaresi.get(rr);
    	  blocksArray[rr][1] = blackSquaresj.get(rr);   	  
      }     
      ImageConverter conv1 = new ImageConverter(imToTraceOver);
      conv1.convertToRGB();         
      Mat copyOriginal = icc.convert(imToTraceOver, Mat.class);
      int xPos;
      int yPos;
      int xDist;
      int yDist;
      int rows;
      int cols;
      float totDist;
      Node initialNode = new Node(0,0);	 
      Node finalNode = new Node(0,0);	
      for (int z = 0; z < iSomaPos.size(); z++) {
    	  finalNode.setRow(iSomaPos.get(z));
    	  finalNode.setCol(jSomaPos.get(z)); 
	      for (int hh = 0; hh < PosI.size(); hh++) {	  
	    	  xDist = Math.abs(PosI.get(hh) - iSomaPos.get(z));	    	  
	    	  yDist = Math.abs(PosJ.get(hh) - jSomaPos.get(z));
	    	  totDist = (float) Math.sqrt((xDist*xDist) + (yDist*yDist));
	    	  if (totDist > Tolerance*2) {
	    		  continue;
	    	  }
	    	  else {
	    		  initialNode.setRow(PosI.get(hh));
	    		  initialNode.setCol(PosJ.get(hh));         
		          rows = Timage.getHeight();
		          cols = Timage.getWidth();
		          AStar aStar = new AStar(rows, cols, initialNode, finalNode);
		          aStar.setBlocks(blocksArray);
		          List<Node> path = aStar.findPath();
		          for (Node node : path) {	
		              xPos = node.getRow();
		              yPos = node.getCol();
		              opencv_imgproc.rectangle(copyOriginal, new Point(xPos-1,yPos-1), new Point(xPos+1,yPos+1), Scalar.YELLOW);		              
	             }
	         }	    	  
	      }
      }     
      ImagePlus temp_image9= mipp.convert(copyOriginal,ImagePlus.class);         
      temp_image9.show();	  	          
  }
  

  
 Vector<Vector<Integer>> findEndpoints(ImagePlus skelDupN) {
	 int neighbors;
	 ImagePlusMatConverter ic = new ImagePlusMatConverter();
	 MatImagePlusConverter mip = new MatImagePlusConverter();   
     Vector<Integer> iPos = new Vector<Integer>();
     Vector<Integer> jPos = new Vector<Integer>();     
     for (int i = 1; i < (skelDupN.getHeight()); i++) {
   	  for (int j = 1; j < (skelDupN.getWidth()-1); j++) {
   		  int[] pix = skelDupN.getPixel(i, j); 
   		  neighbors = 0;			  
   		  if (pix[0] != 0) {   			  
   			  for(int vy = i-1; vy <= i+1; vy++) {                
                     for(int vx = j-1; vx <= j+1; vx++) {                    
                         if(vy == i && vx == j) {                        
                             continue;
                         }
                         else {
                       	  int[] pix1 = skelDupN.getPixel(vy, vx);                        
                             if( pix1[0] != 0 ) {                             
                                 neighbors++;
                             }
                        }
                    }
   			    }  			  
   			  if (neighbors == 1) {
   				  iPos.add(i);
   				  jPos.add(j);   				  
   			   }  			  
   		   } 		  
   	    }   	 	  
     }    
     Mat image_setROI = ic.convert(skelDupN,Mat.class);  
     int i;
     int j;
     for (int k = 0; k < iPos.size(); k++) {
   	  i = iPos.get(k);
   	  j = jPos.get(k);
   	  opencv_imgproc.rectangle(image_setROI, new Point(i-2,j-2), new Point(i+2,j+2), Scalar.YELLOW);
     }
     ImagePlus temp_image2= mip.convert(image_setROI,ImagePlus.class);
     temp_image2.show();
     Vector<Vector<Integer>> endpointPositions = new Vector<Vector<Integer>>();
     endpointPositions.add(iPos);
     endpointPositions.add(jPos);
     return endpointPositions; 	 
 }
 
 
 
 
  
  public KeyPointVector findSomas(Mat neuronCV) {
	  MatImagePlusConverter mipp = new MatImagePlusConverter();
	  opencv_core.Point2fVector soma=new opencv_core.Point2fVector();
  	  KeyPointVector kpv = new opencv_core.KeyPointVector();
  	  opencv_features2d.Feature2D f2d = new opencv_features2d.Feature2D();
  	  opencv_features2d.SimpleBlobDetector.Params parameters=new opencv_features2d.SimpleBlobDetector.Params();
  	  parameters.minThreshold(150);
  	  parameters.maxThreshold(255);
  	  parameters.filterByArea(true);
  	  parameters.minArea(100);
  	  parameters.maxArea(500);
  	  parameters.filterByCircularity(true);
      parameters.minCircularity((float)0.2);
      parameters.filterByConvexity(true);
      parameters.minConvexity((float)0.87);
      parameters.filterByInertia(true);
      parameters.minInertiaRatio((float)0.1);      
  	  f2d=opencv_features2d.SimpleBlobDetector.create(parameters);
      int neuron_radius=10;
      opencv_core.Scalar opencv_white=new opencv_core.Scalar(255);
      opencv_imgproc.medianBlur(neuronCV, neuronCV, 21);
      opencv_core.bitwise_not(neuronCV, neuronCV);
      f2d.detect(neuronCV, kpv);
      opencv_features2d.drawKeypoints(neuronCV, kpv, neuronCV, opencv_white, opencv_features2d.DrawMatchesFlags.DRAW_RICH_KEYPOINTS);      
      return kpv;	  	  
  }
  
  
 
 

  public static void main(final String[] args) throws Exception {
      // Launch ImageJ as usual
      final ImageJ ij = new ImageJ();
      ij.launch(args);
   }
}

