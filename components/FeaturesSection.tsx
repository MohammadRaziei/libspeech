
// Features Section with proper typing
import * as React from "react";
import {FontAwesomeIcon} from "@fortawesome/react-fontawesome";
import {
    faLayerGroup,
    faMicrochip,
    faMicrophone,
    faUser,
    faVolumeUp,
    faWaveSquare
} from "@fortawesome/free-solid-svg-icons";

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

export default function FeaturesSection(){
    const features = [
        {
            icon: <FontAwesomeIcon icon={faMicrochip} className="text-indigo-600 text-3xl" />,
            title: "ONNX Runtime Powered",
            description: "Lightweight AI inference without heavy LibTorch dependency"
        },
        {
            icon: <FontAwesomeIcon icon={faMicrophone} className="text-indigo-600 text-3xl" />,
            title: "Advanced Audio I/O",
            description: "Multi-format support with ultra-low latency processing"
        },
        {
            icon: <FontAwesomeIcon icon={faWaveSquare} className="text-indigo-600 text-3xl" />,
            title: "Feature Extraction",
            description: "MFCC, Spectrogram, and 20+ audio features extracted in real-time"
        },
        {
            icon: <FontAwesomeIcon icon={faVolumeUp} className="text-indigo-600 text-3xl" />,
            title: "Intelligent Noise Reduction",
            description: "AI-powered background noise suppression"
        },
        {
            icon: <FontAwesomeIcon icon={faUser} className="text-indigo-600 text-3xl" />,
            title: "Voice Activity Detection",
            description: "Precision speech segmentation with 95%+ accuracy"
        },
        {
            icon: <FontAwesomeIcon icon={faLayerGroup} className="text-indigo-600 text-3xl" />,
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
