#pragma once
#include<immintrin.h>
class vector4f
{
public:
	inline vector4f(){}
	inline vector4f(float f) : m_value(_mm_set1_ps(f)){}
	inline vector4f(float f0, float f1, float f2, float f3) :m_value(_mm_setr_ps(f0, f1, f2, f3)){}
	inline vector4f(const __m128& rhs) : m_value(rhs){}

	inline vector4f& operator=(const __m128&rhs){
		m_value = rhs;
		return *this;
	}
	inline operator __m128() const { return m_value; }

	//inline vector4f(const vector4f& rhs) : m_value(rhs.m_value){}
	//inline vector4f& operator=(const vector4f& rhs){
	//	m_value = rhs.m_value;
	//	return *this;
	//}

	vector4f();
	~vector4f();


private:
	__m128 m_value;
};

