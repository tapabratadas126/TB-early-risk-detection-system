import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Menu, X, Globe, Volume2, Check } from 'lucide-react';
import { Button } from './ui/button';
import { useLanguage } from '../context/LanguageContext';
import { getTranslation } from '../translations/translations';

export default function Navigation() {
  const [isOpen, setIsOpen] = useState(false);
  const [showCopied, setShowCopied] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const location = useLocation();
  const { language } = useLanguage();
  const t = getTranslation(language);

  const isActive = (path: string) => location.pathname === path;

  const navLinks = [
    { path: '/', label: t.nav.home },
    { path: '/learn', label: t.nav.learn },
    { path: '/prevention', label: t.nav.prevention },
    { path: '/find-help', label: t.nav.findHelp },
    { path: '/privacy', label: t.nav.privacy }
  ];

  const handleCopyLink = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setShowCopied(true);
      setTimeout(() => setShowCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const handleTextToSpeech = () => {
    if (isSpeaking) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
      return;
    }

    // Get only the main content area, excluding navigation
    const mainContent = document.querySelector('main') || 
                       document.querySelector('[role="main"]') ||
                       document.querySelector('.main-content');
    
    let textToRead = '';
    
    if (mainContent) {
      textToRead = mainContent.innerText;
    } else {
      // Fallback: get all text but exclude navigation
      const navElement = document.querySelector('nav');
      const bodyClone = document.body.cloneNode(true) as HTMLElement;
      const navClone = bodyClone.querySelector('nav');
      if (navClone) {
        navClone.remove();
      }
      textToRead = bodyClone.innerText;
    }
    
    // Function to speak with proper voice selection
    const speakText = () => {
      const utterance = new SpeechSynthesisUtterance(textToRead);
      
      // Get available voices
      const voices = window.speechSynthesis.getVoices();
      
      // Set language and try to find appropriate voice
      if (language === 'bn') {
        utterance.lang = 'bn-IN'; // West Bengal Bengali
        // Try to find Bengali voice
        const bengaliVoice = voices.find(voice => 
          voice.lang.includes('bn') || voice.lang.includes('Bengali')
        );
        if (bengaliVoice) {
          utterance.voice = bengaliVoice;
        }
      } else if (language === 'hi') {
        utterance.lang = 'hi-IN'; // Indian Hindi
        // Try to find Hindi voice
        const hindiVoice = voices.find(voice => 
          voice.lang.includes('hi') || voice.lang.includes('Hindi')
        );
        if (hindiVoice) {
          utterance.voice = hindiVoice;
        }
      } else {
        utterance.lang = 'en-IN'; // Indian English
        // Try to find Indian English voice
        const englishVoice = voices.find(voice => 
          voice.lang.includes('en-IN') || (voice.lang.includes('en') && voice.name.includes('India'))
        );
        if (englishVoice) {
          utterance.voice = englishVoice;
        }
      }
      
      utterance.rate = 0.85; // Slower for clear pronunciation
      utterance.pitch = 1;
      utterance.volume = 1;
      
      utterance.onend = () => {
        setIsSpeaking(false);
      };
      
      utterance.onerror = () => {
        setIsSpeaking(false);
      };

      window.speechSynthesis.speak(utterance);
      setIsSpeaking(true);
    };

    // Check if voices are loaded, if not wait for them
    if (window.speechSynthesis.getVoices().length === 0) {
      window.speechSynthesis.addEventListener('voiceschanged', () => {
        speakText();
      }, { once: true });
    } else {
      speakText();
    }
  };

  return (
    <nav className="bg-white border-b border-gray-200 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center">
            <div className="text-xl text-teal-600">
              {t.home.title}
            </div>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-1">
            {navLinks.map((link) => (
              <Link
                key={link.path}
                to={link.path}
                className={`px-3 py-2 rounded-md text-sm transition-colors ${
                  isActive(link.path)
                    ? 'text-teal-600 bg-teal-50'
                    : 'text-gray-600 hover:text-teal-600 hover:bg-gray-50'
                }`}
              >
                {link.label}
              </Link>
            ))}
          </div>

          {/* Accessibility Icons */}
          <div className="hidden md:flex items-center space-x-2">
            <div className="relative">
              <Button 
                variant="ghost" 
                size="sm" 
                className="text-gray-600 hover:text-teal-600"
                onClick={handleCopyLink}
                title="Copy page link"
              >
                {showCopied ? <Check className="h-5 w-5 text-green-600" /> : <Globe className="h-5 w-5" />}
              </Button>
              {showCopied && (
                <div className="absolute top-full right-0 mt-1 px-2 py-1 text-xs bg-gray-900 text-white rounded whitespace-nowrap">
                  Link copied!
                </div>
              )}
            </div>
            <Button 
              variant="ghost" 
              size="sm" 
              className={`text-gray-600 hover:text-teal-600 ${isSpeaking ? 'bg-teal-50 text-teal-600' : ''}`}
              onClick={handleTextToSpeech}
              title={isSpeaking ? "Stop reading" : "Read page aloud"}
            >
              <Volume2 className={`h-5 w-5 ${isSpeaking ? 'animate-pulse' : ''}`} />
            </Button>
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden flex items-center space-x-2">
            <Button 
              variant="ghost" 
              size="sm" 
              className="text-gray-600"
              onClick={handleCopyLink}
            >
              {showCopied ? <Check className="h-5 w-5 text-green-600" /> : <Globe className="h-5 w-5" />}
            </Button>
            <Button 
              variant="ghost" 
              size="sm" 
              className={`text-gray-600 ${isSpeaking ? 'bg-teal-50 text-teal-600' : ''}`}
              onClick={handleTextToSpeech}
            >
              <Volume2 className={`h-5 w-5 ${isSpeaking ? 'animate-pulse' : ''}`} />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsOpen(!isOpen)}
            >
              {isOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
            </Button>
          </div>
        </div>
      </div>

      {/* Mobile Navigation */}
      {isOpen && (
        <div className="md:hidden border-t border-gray-200">
          <div className="px-2 pt-2 pb-3 space-y-1">
            {navLinks.map((link) => (
              <Link
                key={link.path}
                to={link.path}
                onClick={() => setIsOpen(false)}
                className={`block px-3 py-2 rounded-md text-base transition-colors ${
                  isActive(link.path)
                    ? 'text-teal-600 bg-teal-50'
                    : 'text-gray-600 hover:text-teal-600 hover:bg-gray-50'
                }`}
              >
                {link.label}
              </Link>
            ))}
          </div>
        </div>
      )}
    </nav>
  );
}