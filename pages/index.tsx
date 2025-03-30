import Head from 'next/head';
import * as React from 'react';
import Link from 'next/link';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCode, faTrophy, faLightbulb } from '@fortawesome/free-solid-svg-icons';
import { faGithub } from '@fortawesome/free-brands-svg-icons';
import FeaturesSection from '../components/FeaturesSection';

export default function LibSpeechDocumentation() {
  return (
      <main className="bg-gray-50">
        <Head>
          <title>LibSpeech - Lightweight Speech Processing Library</title>
          <meta name="description" content="Lightweight C++ library for speech processing powered by ONNX Runtime" />
        </Head>

        {/* Sponsor Section - Added at the top */}
        <div className="bg-indigo-50 py-4">
          <div className="layout mx-auto px-4 flex flex-col md:flex-row items-center justify-center gap-4">
            <p className="text-sm text-gray-600">Proudly developed by talented engineers at</p>
            <a
                href="http://armantechhub.ir/"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 hover:underline"
            >
              <img
                  src="images/arman-logo.svg"
                  alt="Arman Tech Hub"
                  className="h-8"
              />
              <span className="font-semibold text-indigo-700">Arman Tech Hub</span>
            </a>
          </div>
        </div>

        <section className="layout relative flex min-h-screen flex-col items-center py-12">
          <div className="flex flex-col items-center text-center w-full max-w-4xl">
            <img
                width={200}
                src="https://raw.githubusercontent.com/MohammadRaziei/libspeech/master/docs/logo/libspeech-logo-pods.svg"
                alt="LibSpeech Logo"
                className="mb-6"
            />

            <h1 className="text-4xl font-bold text-gray-900">
              LibSpeech
            </h1>

            <p className="mt-4 text-lg text-gray-600">
              A lightweight, high-performance C++ library for speech processing powered by ONNX Runtime.
              No heavy dependencies like LibTorch required.
            </p>

            {/* Achievement Highlights */}
            <div className="mt-8 bg-white p-6 rounded-xl shadow-sm border border-gray-100 w-full max-w-3xl">
              <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
                <FontAwesomeIcon icon={faTrophy} className="text-yellow-500" />
                Arman Tech Hub Achievements
              </h3>
              <ul className="grid grid-cols-1 md:grid-cols-2 gap-4 text-left">
                {[
                  "Developed cutting-edge AI solutions for Persian language processing",
                  "Pioneers in lightweight, efficient speech technology",
                  "Recipients of multiple national innovation awards",
                  "Home to Iran's top young engineering talent",
                  "Specialists in edge computing and optimized AI deployment"
                ].map((item, index) => (
                    <li key={index} className="flex items-start gap-2">
                      <FontAwesomeIcon icon={faLightbulb} className="text-indigo-500 mt-1 flex-shrink-0" />
                      <span className="text-gray-700">{item}</span>
                    </li>
                ))}
              </ul>
            </div>

            <div className="mt-8 flex flex-wrap justify-center gap-4">
              {[
                {
                  href: "cpp",
                  label: "C++ Documentation",
                  icon: <FontAwesomeIcon icon={faCode} />,
                  className: "bg-blue-600 hover:bg-blue-700"
                },
                {
                  href: "python",
                  label: "Python Documentation",
                  icon: <FontAwesomeIcon icon={faCode} />,
                  className: "bg-green-600 hover:bg-green-700"
                },
                {
                  href: "https://github.com/MohammadRaziei/libspeech",
                  label: "GitHub Repository",
                  icon: <FontAwesomeIcon icon={faGithub} />,
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
          <div className="mt-16 w-full max-w-4xl p-8 rounded-xl shadow-sm">
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
              Created by <a href="https://github.com/MohammadRaziei" className="text-blue-600 hover:underline">Mohammad Raziei</a> and the talented team at <a href="http://armantechhub.ir/" className="text-indigo-600 hover:underline">Arman Tech Hub</a>
            </p>
          </footer>
        </section>
      </main>
  );
}