import { motion } from 'motion/react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Shield, Lock, Eye, Database, UserX, CheckCircle2 } from 'lucide-react';
import Navigation from '../components/Navigation';
import { useLanguage } from '../context/LanguageContext';
import { getTranslation } from '../translations/translations';
import { getComprehensiveTranslation } from '../translations/comprehensive-translations';

export default function PrivacyPage() {
  const { language } = useLanguage();
  const t = getTranslation(language);
  const comprehensiveT = getComprehensiveTranslation(language);

  return (
    <div className="min-h-screen bg-white">
      <Navigation />
      
      <main className="max-w-4xl mx-auto px-4 py-8 sm:py-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <h1 className="text-4xl mb-4 text-gray-900">{t.privacy.title}</h1>
          <p className="text-xl text-gray-600 mb-12">
            {t.privacy.subtitle}
          </p>
        </motion.div>

        {/* Trust Badges */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="mb-8"
        >
          <Card className="bg-green-50 border-green-200">
            <CardContent className="p-6">
              <div className="grid sm:grid-cols-3 gap-6 text-center">
                <div>
                  <div className="w-14 h-14 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-3">
                    <UserX className="h-7 w-7 text-green-600" />
                  </div>
                  <h3 className="text-lg text-green-900 mb-1">{comprehensiveT.privacy.noLoginTitle}</h3>
                  <p className="text-sm text-green-700">{comprehensiveT.privacy.noLoginDesc}</p>
                </div>
                <div>
                  <div className="w-14 h-14 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-3">
                    <Database className="h-7 w-7 text-green-600" />
                  </div>
                  <h3 className="text-lg text-green-900 mb-1">{comprehensiveT.privacy.noStorageTitle}</h3>
                  <p className="text-sm text-green-700">{comprehensiveT.privacy.noStorageDesc}</p>
                </div>
                <div>
                  <div className="w-14 h-14 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-3">
                    <Eye className="h-7 w-7 text-green-600" />
                  </div>
                  <h3 className="text-lg text-green-900 mb-1">{comprehensiveT.privacy.noTrackingTitle}</h3>
                  <p className="text-sm text-green-700">{comprehensiveT.privacy.noTrackingDesc}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <div className="space-y-6">
          {/* How We Handle Your Data */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-teal-100 rounded-full flex items-center justify-center">
                    <Shield className="h-6 w-6 text-teal-600" />
                  </div>
                  {comprehensiveT.privacy.handleDataTitle}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4 text-gray-700">
                <div className="flex items-start gap-3">
                  <CheckCircle2 className="h-5 w-5 text-teal-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <h3 className="text-lg mb-1 text-gray-900">{comprehensiveT.privacy.tempProcessingTitle}</h3>
                    <p>{comprehensiveT.privacy.tempProcessingDesc}</p>
                  </div>
                </div>

                <div className="flex items-start gap-3">
                  <CheckCircle2 className="h-5 w-5 text-teal-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <h3 className="text-lg mb-1 text-gray-900">{comprehensiveT.privacy.noPersonalTitle}</h3>
                    <p>{comprehensiveT.privacy.noPersonalDesc}</p>
                  </div>
                </div>

                <div className="flex items-start gap-3">
                  <CheckCircle2 className="h-5 w-5 text-teal-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <h3 className="text-lg mb-1 text-gray-900">{comprehensiveT.privacy.locationDataTitle}</h3>
                    <p>{comprehensiveT.privacy.locationDataDesc}</p>
                  </div>
                </div>

                <div className="flex items-start gap-3">
                  <CheckCircle2 className="h-5 w-5 text-teal-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <h3 className="text-lg mb-1 text-gray-900">{comprehensiveT.privacy.sessionTitle}</h3>
                    <p>{comprehensiveT.privacy.sessionDesc}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* What We Don't Do */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-red-100 rounded-full flex items-center justify-center">
                    <Lock className="h-6 w-6 text-red-600" />
                  </div>
                  {comprehensiveT.privacy.dontDoTitle}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-gray-700">
                <div className="flex items-start gap-3">
                  <span className="text-red-600 text-xl">✗</span>
                  <p>{comprehensiveT.privacy.dontDo1}</p>
                </div>
                <div className="flex items-start gap-3">
                  <span className="text-red-600 text-xl">✗</span>
                  <p>{comprehensiveT.privacy.dontDo2}</p>
                </div>
                <div className="flex items-start gap-3">
                  <span className="text-red-600 text-xl">✗</span>
                  <p>{comprehensiveT.privacy.dontDo3}</p>
                </div>
                <div className="flex items-start gap-3">
                  <span className="text-red-600 text-xl">✗</span>
                  <p>{comprehensiveT.privacy.dontDo4}</p>
                </div>
                <div className="flex items-start gap-3">
                  <span className="text-red-600 text-xl">✗</span>
                  <p>{comprehensiveT.privacy.dontDo5}</p>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* Your Rights */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
                    <Shield className="h-6 w-6 text-blue-600" />
                  </div>
                  {comprehensiveT.privacy.rightsTitle}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-gray-700">
                <p>{comprehensiveT.privacy.rightsDesc}</p>
                <ul className="space-y-2 ml-4">
                  <li>• {comprehensiveT.privacy.right1}</li>
                  <li>• {comprehensiveT.privacy.right2}</li>
                  <li>• {comprehensiveT.privacy.right3}</li>
                  <li>• {comprehensiveT.privacy.right4}</li>
                </ul>
              </CardContent>
            </Card>
          </motion.div>

          {/* Medical Disclaimer */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.5 }}
          >
            <Card className="border-2 border-amber-200 bg-amber-50">
              <CardHeader>
                <CardTitle className="text-amber-900">{comprehensiveT.privacy.disclaimerTitle}</CardTitle>
              </CardHeader>
              <CardContent className="text-amber-900 space-y-3">
                <p>
                  {comprehensiveT.privacy.disclaimerDesc}
                </p>
                <p>
                  {comprehensiveT.privacy.disclaimerAdvice}
                </p>
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </main>
    </div>
  );
}