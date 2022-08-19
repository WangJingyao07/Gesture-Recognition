#include <opencv2/opencv.hpp>
#include <opencv2/videostab.hpp>
#include <string>
#include <iostream>
 
using namespace std;
using namespace cv;
using namespace cv::videostab;
 
string inputPath = "hand.avi";
string outputPath = "outputVideo.avi";
 
// ��Ƶ�ȶ����
void videoOutput(Ptr<IFrameSource> stabFrames, string outputPath)
{
	VideoWriter writer;
	cv::Mat stabFrame;
	int nframes = 0;
	// �������֡��
	double outputFps = 25;
	// ����������Ƶ֡
	while (!(stabFrame = stabFrames->nextFrame()).empty())
	{
		nframes++;
		// �����Ƶ�ȶ�֡
		if (!outputPath.empty())
		{
			if (!writer.isOpened())
				writer.open(outputPath, VideoWriter::fourcc('X', 'V', 'I', 'D'),
				outputFps, stabFrame.size());
			writer << stabFrame;
		}
		imshow("stabFrame", stabFrame);
		// esc���˳�
		char key = static_cast<char>(waitKey(100));
		if (key == 27)
		{
			cout << endl;
			break;
		}
	}
	std::cout << "nFrames: " << nframes << endl;
	std::cout << "finished " << endl;
}
 
void cacStabVideo(Ptr<IFrameSource> stabFrames, string srcVideoFile)
{
	try
	{
 
		Ptr<VideoFileSource> srcVideo = makePtr<VideoFileSource>(inputPath);
		cout << "frame count: " << srcVideo->count() << endl;
 
		// �˶�����
		double estPara = 0.1;
		Ptr<MotionEstimatorRansacL2> est =
			makePtr<MotionEstimatorRansacL2>(MM_AFFINE);
 
		// Ransac��������
		RansacParams ransac = est->ransacParams();
		ransac.size = 3;
		ransac.thresh = 5;
		ransac.eps = 0.5;
 
		// Ransac����
		est->setRansacParams(ransac);
		est->setMinInlierRatio(estPara);
 
		// Fast�������
		Ptr<FastFeatureDetector> feature_detector =
			FastFeatureDetector::create();
 
		// �˶����ƹؼ���ƥ��
		Ptr<KeypointBasedMotionEstimator> motionEstBuilder =
			makePtr<KeypointBasedMotionEstimator>(est);
 
		// �������������
		motionEstBuilder->setDetector(feature_detector);
		Ptr<IOutlierRejector> outlierRejector = makePtr<NullOutlierRejector>();
		motionEstBuilder->setOutlierRejector(outlierRejector);
 
		// 3-Prepare the stabilizer
		StabilizerBase *stabilizer = 0;
		// first, prepare the one or two pass stabilizer
		bool isTwoPass = 1;
		int radius_pass = 15;
		if (isTwoPass)
		{
			// with a two pass stabilizer
			bool est_trim = true;
			TwoPassStabilizer *twoPassStabilizer = new TwoPassStabilizer();
			twoPassStabilizer->setEstimateTrimRatio(est_trim);
			twoPassStabilizer->setMotionStabilizer(
				makePtr<GaussianMotionFilter>(radius_pass));
			stabilizer = twoPassStabilizer;
		}
		else
		{
			// with an one pass stabilizer
			OnePassStabilizer *onePassStabilizer = new OnePassStabilizer();
			onePassStabilizer->setMotionFilter(
				makePtr<GaussianMotionFilter>(radius_pass));
			stabilizer = onePassStabilizer;
		}
 
		// second, set up the parameters
		int radius = 15;
		double trim_ratio = 0.1;
		bool incl_constr = false;
		stabilizer->setFrameSource(srcVideo);
		stabilizer->setMotionEstimator(motionEstBuilder);
		stabilizer->setRadius(radius);
		stabilizer->setTrimRatio(trim_ratio);
		stabilizer->setCorrectionForInclusion(incl_constr);
		stabilizer->setBorderMode(BORDER_REPLICATE);
		// cast stabilizer to simple frame source interface to read stabilized frames
		stabFrames.reset(dynamic_cast<IFrameSource*>(stabilizer));
		// 4-videoOutput the stabilized frames. The results are showed and saved.
		videoOutput(stabFrames, outputPath);
	}
 
	catch (const exception &e)
	{
		cout << "error: " << e.what() << endl;
		stabFrames.release();
	}
}
 
int main(int argc, char* argv[])
{
	Ptr<IFrameSource> stabFrames;
	// ���������Ƶ׼��
 
	cacStabVideo(stabFrames, inputPath);
	stabFrames.release();
 
	return 0;
}
 

