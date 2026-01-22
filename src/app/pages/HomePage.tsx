import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, useInView } from 'motion/react';
import { Button } from '../components/ui/button';
import { Card, CardContent } from '../components/ui/card';
import { ImageWithFallback } from '../components/figma/ImageWithFallback';
import { CheckCircle2, ArrowRight, Shield, Clock, Users, AlertCircle, Languages } from 'lucide-react';
import Navigation from '../components/Navigation';
import { useLanguage, Language } from '../context/LanguageContext';
import { getTranslation } from '../translations/translations';
import { getComprehensiveTranslation } from '../translations/comprehensive-translations';

const AnimatedSection = ({ children, delay = 0 }: { children: React.ReactNode; delay?: number }) => {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, amount: 0.2 });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 30 }}
      animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
      transition={{ duration: 0.5, delay }}
    >
      {children}
    </motion.div>
  );
};

export default function HomePage() {
  const navigate = useNavigate();
  const [scrollY, setScrollY] = useState(0);
  const { language, setLanguage } = useLanguage();
  const t = getTranslation(language);
  const comprehensiveT = getComprehensiveTranslation(language);

  useEffect(() => {
    const handleScroll = () => setScrollY(window.scrollY);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div className="min-h-screen bg-white">
      <Navigation />
      
      <main>
        {/* Hero Section - Story Driven */}
        <section className="relative bg-gradient-to-b from-teal-50/40 to-white">
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 pt-16 pb-20">
            {/* Language Switcher */}
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4 }}
              className="mb-6 flex justify-center"
            >
              <div className="inline-flex items-center gap-3 bg-white border-2 border-teal-100 rounded-full px-6 py-3 shadow-md">
                <Languages className="h-5 w-5 text-teal-600" />
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    onClick={() => setLanguage('en')}
                    className={`rounded-full px-6 py-2 transition-all ${
                      language === 'en'
                        ? 'bg-teal-600 text-white hover:bg-teal-700'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    English
                  </Button>
                  <Button
                    size="sm"
                    onClick={() => setLanguage('bn')}
                    className={`rounded-full px-6 py-2 transition-all ${
                      language === 'bn'
                        ? 'bg-teal-600 text-white hover:bg-teal-700'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    বাংলা
                  </Button>
                  <Button
                    size="sm"
                    onClick={() => setLanguage('hi')}
                    className={`rounded-full px-6 py-2 transition-all ${
                      language === 'hi'
                        ? 'bg-teal-600 text-white hover:bg-teal-700'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    हिंदी
                  </Button>
                </div>
              </div>
            </motion.div>

            {/* Disclaimer Banner */}
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: 0.1 }}
              className="mb-6"
            >
              <div className="bg-blue-50 border border-blue-200 rounded-lg px-4 py-3 text-center">
                <p className="text-sm text-blue-900">
                  <strong>{t.home.disclaimer}:</strong> {t.home.disclaimerText}
                </p>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="text-center"
            >
              {/* Hero Headline */}
              <h1 className="text-4xl sm:text-5xl md:text-6xl mb-4 text-gray-900">
                {t.home.title}
              </h1>
              
              {/* Purpose Line */}
              <p className="text-xl sm:text-2xl text-gray-600 mb-8 max-w-2xl mx-auto">
                {t.home.description}
              </p>

              {/* Hero Image */}
              <div className="mb-10 mx-auto max-w-xl">
                <ImageWithFallback
                  src="https://images.unsplash.com/photo-1758691462126-2ee47c8bf9e7?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&q=80&w=1080"
                  alt="Healthcare"
                  className="rounded-2xl shadow-lg w-full h-64 sm:h-80 object-cover"
                />
              </div>

              {/* ONE Primary CTA */}
              <Button
                size="lg"
                onClick={() => navigate('/assessment')}
                className="bg-teal-600 hover:bg-teal-700 text-white px-12 py-7 text-lg rounded-full shadow-lg hover:shadow-xl transition-all"
              >
                {t.home.cta}
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>

              {/* Trust Badge */}
              <div className="mt-8 inline-flex items-center gap-2 text-sm text-gray-500">
                <Shield className="h-4 w-4 text-teal-600" />
                <span>{t.home.badge}</span>
              </div>
            </motion.div>
          </div>
        </section>

        {/* Why Early Detection Matters */}
        <section className="py-16 sm:py-20 bg-white">
          <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
            <AnimatedSection>
              <div className="text-center mb-12">
                <h2 className="text-3xl sm:text-4xl mb-4 text-gray-900">
                  {comprehensiveT.home.whyDetection}
                </h2>
                <p className="text-lg text-gray-600 max-w-2xl mx-auto">
                  {comprehensiveT.home.whyDetectionDesc}
                </p>
              </div>
            </AnimatedSection>

            <div className="grid sm:grid-cols-3 gap-6 sm:gap-8">
              <AnimatedSection delay={0.1}>
                <Card className="border-teal-100 hover:shadow-md transition-shadow">
                  <CardContent className="pt-8 pb-8 text-center">
                    <div className="w-16 h-16 bg-teal-100 rounded-full flex items-center justify-center mx-auto mb-4">
                      <CheckCircle2 className="h-8 w-8 text-teal-600" />
                    </div>
                    <h3 className="text-xl mb-3 text-gray-900">{comprehensiveT.home.preventTitle}</h3>
                    <p className="text-gray-600">
                      {comprehensiveT.home.preventDesc}
                    </p>
                  </CardContent>
                </Card>
              </AnimatedSection>

              <AnimatedSection delay={0.2}>
                <Card className="border-blue-100 hover:shadow-md transition-shadow">
                  <CardContent className="pt-8 pb-8 text-center">
                    <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                      <Clock className="h-8 w-8 text-blue-600" />
                    </div>
                    <h3 className="text-xl mb-3 text-gray-900">{comprehensiveT.home.fasterTitle}</h3>
                    <p className="text-gray-600">
                      {comprehensiveT.home.fasterDesc}
                    </p>
                  </CardContent>
                </Card>
              </AnimatedSection>

              <AnimatedSection delay={0.3}>
                <Card className="border-purple-100 hover:shadow-md transition-shadow">
                  <CardContent className="pt-8 pb-8 text-center">
                    <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
                      <Users className="h-8 w-8 text-purple-600" />
                    </div>
                    <h3 className="text-xl mb-3 text-gray-900">{comprehensiveT.home.reduceTitle}</h3>
                    <p className="text-gray-600">
                      {comprehensiveT.home.reduceDesc}
                    </p>
                  </CardContent>
                </Card>
              </AnimatedSection>
            </div>
          </div>
        </section>

        {/* How It Works - 3 Simple Steps */}
        <section className="py-16 sm:py-20 bg-gradient-to-b from-gray-50 to-white">
          <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
            <AnimatedSection>
              <div className="text-center mb-12">
                <h2 className="text-3xl sm:text-4xl mb-4 text-gray-900">
                  {comprehensiveT.home.howWorksTitle}
                </h2>
                <p className="text-lg text-gray-600">
                  {comprehensiveT.home.howWorksDesc}
                </p>
              </div>
            </AnimatedSection>

            <div className="space-y-8">
              {[
                {
                  step: comprehensiveT.home.step1Number,
                  title: comprehensiveT.home.step1Title,
                  desc: comprehensiveT.home.step1Desc,
                  color: 'teal'
                },
                {
                  step: comprehensiveT.home.step2Number,
                  title: comprehensiveT.home.step2Title,
                  desc: comprehensiveT.home.step2Desc,
                  color: 'blue'
                },
                {
                  step: comprehensiveT.home.step3Number,
                  title: comprehensiveT.home.step3Title,
                  desc: comprehensiveT.home.step3Desc,
                  color: 'purple'
                }
              ].map((item, index) => (
                <AnimatedSection key={index} delay={index * 0.1}>
                  <div className="flex items-start gap-6 bg-white p-6 rounded-xl border border-gray-200 hover:shadow-md transition-shadow">
                    <div className={`flex-shrink-0 w-12 h-12 sm:w-14 sm:h-14 bg-${item.color}-100 rounded-full flex items-center justify-center`}>
                      <span className={`text-xl sm:text-2xl text-${item.color}-600`}>{item.step}</span>
                    </div>
                    <div className="flex-1">
                      <h3 className="text-xl sm:text-2xl mb-2 text-gray-900">{item.title}</h3>
                      <p className="text-gray-600">{item.desc}</p>
                    </div>
                  </div>
                </AnimatedSection>
              ))}
            </div>
          </div>
        </section>

        {/* User Journey / Impact Section */}
        <section className="py-16 sm:py-20 bg-teal-50/30">
          <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
            <AnimatedSection>
              <div className="text-center mb-12">
                <h2 className="text-3xl sm:text-4xl mb-4 text-gray-900">
                  {comprehensiveT.home.journeyTitle}
                </h2>
                <p className="text-lg text-gray-600">
                  {comprehensiveT.home.journeyDesc}
                </p>
              </div>
            </AnimatedSection>

            <AnimatedSection delay={0.2}>
              <div className="relative">
                <div className="absolute top-0 left-1/2 transform -translate-x-1/2 h-full w-1 bg-teal-200 hidden sm:block"></div>
                
                <div className="space-y-12">
                  {[
                    { title: comprehensiveT.home.journey1, icon: '📱', desc: comprehensiveT.home.journey1Desc },
                    { title: comprehensiveT.home.journey2, icon: '📊', desc: comprehensiveT.home.journey2Desc },
                    { title: comprehensiveT.home.journey3, icon: '🏥', desc: comprehensiveT.home.journey3Desc },
                    { title: comprehensiveT.home.journey4, icon: '✅', desc: comprehensiveT.home.journey4Desc }
                  ].map((item, index) => (
                    <div key={index} className="relative flex items-center gap-6">
                      <div className="hidden sm:block absolute left-1/2 transform -translate-x-1/2 w-12 h-12 bg-white rounded-full border-4 border-teal-200 flex items-center justify-center z-10">
                        <span className="text-2xl">{item.icon}</span>
                      </div>
                      <div className={`flex-1 ${index % 2 === 0 ? 'sm:text-right sm:pr-8' : 'sm:pl-8 sm:ml-auto'}`}>
                        <Card className="bg-white border-teal-100">
                          <CardContent className="p-6">
                            <div className="flex items-center gap-3 sm:hidden mb-2">
                              <span className="text-2xl">{item.icon}</span>
                            </div>
                            <h3 className="text-lg mb-2 text-gray-900">{item.title}</h3>
                            <p className="text-gray-600">{item.desc}</p>
                          </CardContent>
                        </Card>
                      </div>
                      <div className="flex-1 hidden sm:block"></div>
                    </div>
                  ))}
                </div>
              </div>
            </AnimatedSection>
          </div>
        </section>

        {/* Final CTA Section */}
        <section className="py-16 sm:py-20 bg-white">
          <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <AnimatedSection>
              <div className="bg-gradient-to-br from-teal-50 to-blue-50 rounded-2xl p-8 sm:p-12">
                <h2 className="text-3xl sm:text-4xl mb-6 text-gray-900">
                  {comprehensiveT.home.readyTitle}
                </h2>
                <p className="text-lg text-gray-600 mb-8 max-w-2xl mx-auto">
                  {comprehensiveT.home.readyDesc}
                </p>
                
                <Button
                  size="lg"
                  onClick={() => navigate('/assessment')}
                  className="bg-teal-600 hover:bg-teal-700 text-white px-12 py-7 text-lg rounded-full shadow-lg hover:shadow-xl transition-all"
                >
                  {comprehensiveT.home.startButton}
                  <ArrowRight className="ml-2 h-5 w-5" />
                </Button>

                <div className="mt-6 flex items-center justify-center gap-2 text-sm text-gray-500">
                  <AlertCircle className="h-4 w-4" />
                  <span>{comprehensiveT.home.footerDisclaimer}</span>
                </div>
              </div>
            </AnimatedSection>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="py-8 bg-gray-50 border-t border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <p className="text-gray-600 mb-2">
            {comprehensiveT.home.footerSubtitle}
          </p>
          <p className="text-sm text-gray-500">
            {comprehensiveT.home.footerAdvice}
          </p>
        </div>
      </footer>
    </div>
  );
}