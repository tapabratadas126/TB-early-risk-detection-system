import { motion } from 'motion/react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Stethoscope, AlertCircle, Users, Pill, CheckCircle2, XCircle } from 'lucide-react';
import Navigation from '../components/Navigation';
import { useLanguage } from '../context/LanguageContext';
import { getTranslation } from '../translations/translations';
import { getComprehensiveTranslation } from '../translations/comprehensive-translations';

export default function LearnPage() {
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
          <h1 className="text-4xl mb-4 text-gray-900">{t.learn.title}</h1>
          <p className="text-xl text-gray-600 mb-12">
            {t.learn.subtitle}
          </p>
        </motion.div>

        <div className="space-y-8">
          {/* What is TB */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-teal-100 rounded-full flex items-center justify-center">
                    <Stethoscope className="h-6 w-6 text-teal-600" />
                  </div>
                  {t.learn.whatIs}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-gray-700">
                <p>
                  {t.learn.whatIsDesc}
                </p>
              </CardContent>
            </Card>
          </motion.div>

          {/* Latent vs Active TB */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
                    <AlertCircle className="h-6 w-6 text-blue-600" />
                  </div>
                  {comprehensiveT.learn.latentTitle}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid sm:grid-cols-2 gap-6">
                  <div className="space-y-3">
                    <h3 className="text-lg text-gray-900">{comprehensiveT.learn.latentSubtitle}</h3>
                    <ul className="space-y-2 text-gray-700">
                      <li className="flex items-start gap-2">
                        <CheckCircle2 className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                        <span>{comprehensiveT.learn.latent1}</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <CheckCircle2 className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                        <span>{comprehensiveT.learn.latent2}</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <CheckCircle2 className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                        <span>{comprehensiveT.learn.latent3}</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <CheckCircle2 className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                        <span>{comprehensiveT.learn.latent4}</span>
                      </li>
                    </ul>
                  </div>

                  <div className="space-y-3">
                    <h3 className="text-lg text-gray-900">{comprehensiveT.learn.activeSubtitle}</h3>
                    <ul className="space-y-2 text-gray-700">
                      <li className="flex items-start gap-2">
                        <AlertCircle className="h-5 w-5 text-amber-600 flex-shrink-0 mt-0.5" />
                        <span>{comprehensiveT.learn.active1}</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <AlertCircle className="h-5 w-5 text-amber-600 flex-shrink-0 mt-0.5" />
                        <span>{comprehensiveT.learn.active2}</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <AlertCircle className="h-5 w-5 text-amber-600 flex-shrink-0 mt-0.5" />
                        <span>{comprehensiveT.learn.active3}</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <AlertCircle className="h-5 w-5 text-amber-600 flex-shrink-0 mt-0.5" />
                        <span>{comprehensiveT.learn.active4}</span>
                      </li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* How TB Spreads */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center">
                    <Users className="h-6 w-6 text-purple-600" />
                  </div>
                  {comprehensiveT.learn.spreadTitle}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  <div>
                    <h3 className="text-lg mb-3 text-gray-900 flex items-center gap-2">
                      <CheckCircle2 className="h-5 w-5 text-red-600" />
                      {comprehensiveT.learn.spreadYesTitle}
                    </h3>
                    <ul className="space-y-2 text-gray-700 ml-7">
                      <li>• {comprehensiveT.learn.spreadYes1}</li>
                      <li>• {comprehensiveT.learn.spreadYes2}</li>
                      <li>• {comprehensiveT.learn.spreadYes3}</li>
                    </ul>
                  </div>

                  <div>
                    <h3 className="text-lg mb-3 text-gray-900 flex items-center gap-2">
                      <XCircle className="h-5 w-5 text-green-600" />
                      {comprehensiveT.learn.spreadNoTitle}
                    </h3>
                    <ul className="space-y-2 text-gray-700 ml-7">
                      <li>• {comprehensiveT.learn.spreadNo1}</li>
                      <li>• {comprehensiveT.learn.spreadNo2}</li>
                      <li>• {comprehensiveT.learn.spreadNo3}</li>
                      <li>• {comprehensiveT.learn.spreadNo4}</li>
                      <li>• {comprehensiveT.learn.spreadNo5}</li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* Treatment */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center">
                    <Pill className="h-6 w-6 text-green-600" />
                  </div>
                  {comprehensiveT.learn.treatmentTitle}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                  <p className="text-green-900">
                    {comprehensiveT.learn.treatmentGoodNews}
                  </p>
                </div>

                <div className="space-y-3 text-gray-700">
                  <h3 className="text-lg text-gray-900">{comprehensiveT.learn.treatmentBasics}</h3>
                  <ul className="space-y-2 ml-4">
                    <li>• {comprehensiveT.learn.treatment1}</li>
                    <li>• {comprehensiveT.learn.treatment2}</li>
                    <li>• {comprehensiveT.learn.treatment3}</li>
                    <li>• {comprehensiveT.learn.treatment4}</li>
                  </ul>
                </div>

                <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 mt-4">
                  <p className="text-amber-900">
                    <strong>{comprehensiveT.learn.treatmentImportant}</strong> {comprehensiveT.learn.treatmentWarning}
                  </p>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </main>
    </div>
  );
}