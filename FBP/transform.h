#define PI 3.14159265359f
void para_setup_fan_curve( float D, float R, int nViews, float dRange, float dAngle, int nCols, int nX, int nY, float rFOV, float dx, float dy, float xmin, float ymin, float dCol, float dShift, float dLeft, int nSamp );
void backproject_fan_curve( float * volume, const float * projection, const float * anglePos);
