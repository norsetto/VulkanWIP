#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec4 position;
layout(location = 1) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConsts {
	float time;
} pushConsts;

vec3 mod289(vec3 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 mod289(vec4 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute(vec4 x) {
     return mod289(((x*34.0)+1.0)*x);
}

vec4 taylorInvSqrt(vec4 r)
{
  return 1.79284291400159 - 0.85373472095314 * r;
}

/*
 Description : Array and textureless GLSL 2D/3D/4D simplex 
               noise functions.
      Author : Ian McEwan, Ashima Arts.
  Maintainer : ijm
     Lastmod : 20110822 (ijm)
     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
               Distributed under the MIT License. See LICENSE file.
               https://github.com/ashima/webgl-noise
*/ 

float snoise(vec3 v)
  { 
  const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

// First corner
  vec3 i  = floor(v + dot(v, C.yyy) );
  vec3 x0 =   v - i + dot(i, C.xxx) ;

// Other corners
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min( g.xyz, l.zxy );
  vec3 i2 = max( g.xyz, l.zxy );

  //   x0 = x0 - 0.0 + 0.0 * C.xxx;
  //   x1 = x0 - i1  + 1.0 * C.xxx;
  //   x2 = x0 - i2  + 2.0 * C.xxx;
  //   x3 = x0 - 1.0 + 3.0 * C.xxx;
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
  vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

// Permutations
  i = mod289(i); 
  vec4 p = permute( permute( permute( 
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

// Gradients: 7x7 points over a square, mapped onto an octahedron.
// The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
  float n_ = 0.142857142857; // 1.0/7.0
  vec3  ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );

  //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
  //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);

//Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

// Mix final noise value
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                dot(p2,x2), dot(p3,x3) ) );
  }

void main() {

	float size = 0.75; //The higher, the smaller the perturbation

	//Perturb the coords with three components of noise
	vec3 P = position.xyz;
	vec3 uvw = P + 0.1*vec3(snoise(P + vec3(  0.0,   0.0, pushConsts.time)),
							snoise(P + vec3( 43.0,  17.0, pushConsts.time)),
							snoise(P + vec3(-17.0, -43.0, pushConsts.time)));

	//Six components of noise in a fractal sum (persistence = 0.5)
	float n = snoise(uvw - vec3(0.0, 0.0, pushConsts.time));
	n += 0.50000 * snoise(uvw * 2.00 - vec3(0.0, 0.0, pushConsts.time * 1.4));
	n += 0.25000 * snoise(uvw * 4.00 - vec3(0.0, 0.0, pushConsts.time * 2.0));
	n += 0.12500 * snoise(uvw * 8.00 - vec3(0.0, 0.0, pushConsts.time * 2.8));
	n += 0.06250 * snoise(uvw * 16.0 - vec3(0.0, 0.0, pushConsts.time * 4.0));
	n += 0.03125 * snoise(uvw * 32.0 - vec3(0.0, 0.0, pushConsts.time * 5.6));
	n *= size;
	n = abs(sin(n)); //Marble
//  n = fract(n); //Wood
    
	/*
     Write final color to the framebuffer:
       n >> 0  -> the perturbation color (the 3rd one)
       n ~ 0.5 -> the "surrounding" of the perturbation (the 2nd color)
       n >> 1  -> the background color (the 1st one)
    */
	outColor = n * n * vec4(fragColor, 1.0) + 2.0 * n * (1.0 - n) * vec4(0.25, 0.25, 0.25, 0.5) + (1.0 - n) * (1.0 - n) * vec4(1.0, 1.0, 0.0, 0.25);
}