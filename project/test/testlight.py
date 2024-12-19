#include <opencv2\opencv.hpp>
#include <math.h>

/********************************************************************************
单尺度Retinex图像增强程序
src为待处理图像
sigma为高斯模糊标准差
*********************************************************************************/
void SingleScaleRetinex (const cv::Mat &src, cv::Mat &dst, int sigma);

/********************************************************************************
多尺度Retinex图像增强程序
src为待处理图像
k为尺度参数
w为权重参数
sigma为高斯模糊标准差
*********************************************************************************/
void MultiScaleRetinex  (const cv::Mat &src, cv::Mat &dst, int k, std::vector<double> w, std::vector<double> sigmas);

/********************************************************************************
多尺度Retinex图像增强程序
src     为待处理图像
k       为尺度参数
w       为权重参数
sigma   为高斯模糊标准差
alpha   增益常数
beta    受控的非线性强度
gain    增益
offset  偏差
*********************************************************************************/
void MultiScaleRetinexCR(const cv::Mat &src, cv::Mat &dst, int k, std::vector<double> w, std::vector<double> sigmas, int alpha , int beta , int gain, int offset);

int main()
{

	cv::Mat src = cv::imread("D:\\CPP_Project\\OpenCV410\\OpenCV410\\image\\test_MSRCR.jpg", cv::IMREAD_COLOR);
	if (src.empty()) 
	{
		std::cout << "The image is empty" << std::endl;
		
		return 1;
	}
		
	cv::Mat res1 ,res2, res3;
	

	int k = 3;

	std::vector < double > w(k);
	w[0] = 0.333333333;
	w[1] = 0.333333333;
	w[2] = 0.333333334;

	std::vector<double> s(k);
	s[0] = 15;
	s[1] = 80;
	s[2] = 200;


	SingleScaleRetinex (src, res1, 200);

	MultiScaleRetinex  (src, res2, k, w, s);

	MultiScaleRetinexCR(src, res3, k, w, s, 1, 1, 1, 0);

	cv::imshow("org", src);
	cv::imshow("res1", res1);
	cv::imshow("res2", res2);
	cv::imshow("res3", res3);
	cv::waitKey(0);
	
	return 0;
}

void SingleScaleRetinex(const cv::Mat &src, cv::Mat &dst, int sigma)
{
	cv::Mat doubleI, gaussianI, logI, logGI, logR;

	src.convertTo(doubleI, CV_64FC3, 1.0, 1.0);                    //转换范围，所有图像元素增加1.0保证cvlog正常
	cv::GaussianBlur(doubleI, gaussianI, cv::Size(0,0), sigma);    //SSR算法的核心之一，高斯模糊，当size为零时将通过sigma自动进行计算
	cv::log(doubleI, logI);
	cv::log(gaussianI, logGI);
	logR = logI - logGI;                                           //Retinex公式，Log(R(x,y))=Log(I(x,y))-Log(Gauss(I(x,y)))
	cv::normalize(logR, dst, 0, 255, cv::NORM_MINMAX,CV_8UC3);     //SSR算法的核心之二,线性量化 (似乎在量化的时候没有谁会将 Log[R(x,y)]进行Exp函数的运算而直接得到R(x,y))
}


void MultiScaleRetinex(const cv::Mat &src, cv::Mat &dst, int k, std::vector<double> w, std::vector<double> sigmas) 
{
	cv::Mat doubleI, logI;
	cv::Mat logR = cv::Mat::zeros(src.size(),CV_64FC3);

	src.convertTo(doubleI, CV_64FC3, 1.0, 1.0);                    //转换范围，所有图像元素增加1.0保证cvlog正常
	cv::log(doubleI, logI);

	for (int i = 0; i < k; i++)
	{//Retinex公式，Log(R(x,y)) += w_k(Log(I(x,y))-Log(Gauss_k(I(x,y))))
		cv::Mat tempGI;
		cv::GaussianBlur(doubleI, tempGI, cv::Size(0, 0), sigmas[i]);
		cv::Mat templogGI;
		cv::log(tempGI, templogGI);
		logR += w[i]*(logI - templogGI);
	}

	cv::normalize(logR, dst, 0, 255, cv::NORM_MINMAX, CV_8UC3);  //SSR算法的核心之二,线性量化 (似乎在量化的时候没有谁会将 Log[R(x,y)]进行Exp函数的运算而直接得到R(x,y))
}

//
void MultiScaleRetinexCR(const cv::Mat &src, cv::Mat &dst, int k, std::vector<double> w, std::vector<double> sigmas, int alpha, int beta, int gain, int offset)
{
	

	cv::Mat doubleIl, logI;
	cv::Mat logMSR = cv::Mat::zeros(src.size(), CV_64FC3);

	src.convertTo(doubleIl, CV_64FC3, 1.0, 1.0);                    //转换范围，所有图像元素增加1.0保证cvlog正常
	cv::log(doubleIl, logI);

	for (int i = 0; i < k; i++)
	{//Retinex公式，Log(R(x,y)) += w_k(Log(I(x,y))-Log(Gauss_k(I(x,y))))
		cv::Mat tempGI;
		cv::GaussianBlur(doubleIl, tempGI, cv::Size(0, 0), sigmas[i]);
		cv::Mat templogGI;
		cv::log(tempGI, templogGI);
		logMSR += w[i] * (logI - templogGI);
	}


	std::vector<cv::Mat> logMSRc(3);
	cv::split(logMSR, logMSRc);

	cv::Mat doubleI;
	src.convertTo(doubleI, CV_64FC3);

	std::vector<cv::Mat> doubleIc(3);
	cv::split(doubleI, doubleIc);


	cv::Mat sumDoubleIc =cv::Mat::zeros(doubleI.size(),CV_64FC1);
	for (int i = 0; i < doubleI.rows; i++)
	{
		for (int j = 0; j < doubleI.cols; j++)
		{
			sumDoubleIc.ptr<double>(i)[j] = doubleI.ptr<cv::Vec3d>(i)[j][0] + doubleI.ptr<cv::Vec3d>(i)[j][1] + doubleI.ptr<cv::Vec3d>(i)[j][2];
		}
	}


	std::vector<cv::Mat> divideDoubleIc(3);
	std::vector<cv::Mat> Cc(3);
	std::vector<cv::Mat> MSRCRc(3);
	cv::Mat tempResult;

	for (int i = 0; i < 3; i++)
	{
		cv::divide(doubleIc[i], sumDoubleIc, divideDoubleIc[i]);
		divideDoubleIc[i].convertTo(divideDoubleIc[i], CV_64FC1, 1.0, 1.0);
		cv::log(alpha * divideDoubleIc[i], Cc[i]);
		Cc[i] *= beta;
		MSRCRc[i] = Cc[i].mul(logMSRc[i]);
		MSRCRc[i] = gain * MSRCRc[i] + offset;
	}


	cv::merge(MSRCRc, tempResult);
	cv::normalize(tempResult, dst, 0, 255, cv::NORM_MINMAX, CV_8UC3);
}


————————————————

                            版权声明：本文为博主原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接和本声明。
                        
原文链接：https://blog.csdn.net/wsp_1138886114/article/details/83096109