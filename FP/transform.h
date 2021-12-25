#define PI 3.14159265359f
void para_setup_fan_curve(float D, float R, int nViews, float dRange, float dAngle, int nCols, int nX, int nY, float rFOV, float dx, float dy, float xmin, float ymin, float dCol, float dLeft, int nSamp);
void project_fan_curve(const float * volume, float * projection, const float * anglePos);
