// import getConfig from 'next/config';

import Layout from '@/components/Layout';
import Head from 'next/head';
import * as React from 'react';
import Link from 'next/link';
import { FiCpu, FiMic, FiActivity, FiVolume2, FiUser, FiLayers, FiCode, FiGithub } from 'react-icons/fi';



// const { publicRuntimeConfig } = getConfig();
// const { name } = publicRuntimeConfig.site;





/**
 * SVGR Support
 * Caveat: No React Props Type.
 *
 * You can override the next-env if the type is important to you
 * @see https://stackoverflow.com/questions/68103844/how-to-override-next-js-svg-module-declaration
 */

// !STARTERCONF -> Select !STARTERCONF and CMD + SHIFT + F
// Before you begin editing, follow all comments with `STARTERCONF`,
// to customize the default configuration.

// Features Section with proper typing
const FeatureCard = ({ icon, title, description }: {
  icon: React.ReactNode;
  title: string;
  description: string;
}) => (
    <div className="group bg-white p-8 rounded-2xl border border-gray-100 hover:border-indigo-100 transition-all duration-300 hover:shadow-lg">
      <div className="flex items-start gap-4 mb-4">
        <div className="p-3 bg-indigo-50 rounded-lg group-hover:bg-indigo-100 transition-colors">
          {icon}
        </div>
        <h3 className="text-xl font-semibold text-gray-900 mt-1">
          {title}
        </h3>
      </div>
      <p className="text-gray-600 pl-[60px]">
        {description}
      </p>
    </div>
);

const FeaturesSection = () => {
  const features = [
    {
      icon: <FiCpu className="text-indigo-600 text-3xl" />,
      title: "ONNX Runtime Powered",
      description: "Lightweight AI inference without heavy LibTorch dependency"
    },
    {
      icon: <FiMic className="text-indigo-600 text-3xl" />,
      title: "Advanced Audio I/O",
      description: "Multi-format support with ultra-low latency processing"
    },
    {
      icon: <FiActivity className="text-indigo-600 text-3xl" />,
      title: "Feature Extraction",
      description: "MFCC, Spectrogram, and 20+ audio features extracted in real-time"
    },
    {
      icon: <FiVolume2 className="text-indigo-600 text-3xl" />,
      title: "Intelligent Noise Reduction",
      description: "AI-powered background noise suppression"
    },
    {
      icon: <FiUser className="text-indigo-600 text-3xl" />,
      title: "Voice Activity Detection",
      description: "Precision speech segmentation with 95%+ accuracy"
    },
    {
      icon: <FiLayers className="text-indigo-600 text-3xl" />,
      title: "Cross-Platform",
      description: "Seamless performance across Windows, Linux & macOS"
    }
  ];

  return (
      <div style={{"marginTop":70}} className="w-full max-w-5xl">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-3">
            Powerful Features
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Engineered for high-performance speech processing with minimal dependencies
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
              <FeatureCard key={index} {...feature} />
          ))}
        </div>
      </div>
  );
};

export default function LibSpeechDocumentation() {
  return (
      <main className="bg-gray-50">
        <Head>
          <title>LibSpeech - Lightweight Speech Processing Library</title>
          <meta name="description" content="Lightweight C++ library for speech processing powered by ONNX Runtime" />
        </Head>

        <section className="layout relative flex min-h-screen flex-col items-center py-12">
          <div className="flex flex-col items-center text-center">
            <img
                width={200}
                src="https://raw.githubusercontent.com/MohammadRaziei/libspeech/master/docs/logo/libspeech-logo-pods.svg"
                alt="LibSpeech Logo"
                className="mb-6"
            />

            <h1 className="text-4xl font-bold text-gray-900">
              LibSpeech
            </h1>

            <p className="mt-4 max-w-2xl text-lg text-gray-600">
              A lightweight, high-performance C++ library for speech processing powered by ONNX Runtime.
              No heavy dependencies like LibTorch required.
            </p>



            <div className="mt-8 flex flex-wrap justify-center gap-4">
              {[
                {
                  href: "cpp",
                  label: "C++ Documentation",
                  icon: <FiCode />,
                  className: "bg-blue-600 hover:bg-blue-700"
                },
                {
                  href: "python",
                  label: "Python Bindings",
                  icon: <FiCode />,
                  className: "bg-green-600 hover:bg-green-700"
                },
                {
                  href: "https://github.com/MohammadRaziei/libspeech",
                  label: "GitHub Repository",
                  icon: <FiGithub />,
                  className: "bg-gray-800 hover:bg-gray-900",
                  external: true
                }
              ].map((button, index) => {
                const commonClasses = "flex items-center gap-2 rounded-lg px-6 py-3 text-white transition-colors";

                return button.external ? (
                    <a
                        key={index}
                        href={button.href}
                        target="_blank"
                        rel="noopener noreferrer"
                        className={`${commonClasses} ${button.className}`}
                    >
                      {button.icon} {button.label}
                    </a>
                ) : (
                    <Link
                        key={index}
                        href={button.href}
                        className={`${commonClasses} ${button.className}`}
                    >
                      {button.icon} {button.label}
                    </Link>
                );
              })}
            </div>
          </div>






          <FeaturesSection />











          {/* Quick Start Section */}
          <div className="mt-16 w-full max-w-4xl  p-8 rounded-xl shadow-sm">
            <h2 className="text-2xl font-semibold text-gray-800 mb-6">
              Quick Start
            </h2>

            <div className="space-y-6">
              <div>
                <h3 className="font-medium text-gray-800 mb-2">C++ Installation</h3>
                <pre className="bg-gray-100 p-4 rounded-md overflow-x-auto">
                <code>
                  {`# Clone the repository
git clone https://github.com/MohammadRaziei/libspeech
cd libspeech

# Build and install
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make install`}
                </code>
              </pre>
              </div>

              <div>
                <h3 className="font-medium text-gray-800 mb-2">Python Installation</h3>
                <pre className="bg-gray-100 p-4 rounded-md overflow-x-auto">
                <code>
                  {`# Install from PyPI
pip install libspeech

# Or install from source
pip install git+https://github.com/MohammadRaziei/libspeech.git`}
                </code>
              </pre>
              </div>
            </div>
          </div>

          <footer className="mt-auto pt-12 text-center text-gray-500 text-sm">
            <p>© {new Date().getFullYear()} LibSpeech. All rights reserved.</p>
            <p className="mt-1">
              Created by <a href="https://github.com/MohammadRaziei" className="text-blue-600 hover:underline">Mohammad Raziei</a>
            </p>
          </footer>
        </section>
      </main>
  );
}