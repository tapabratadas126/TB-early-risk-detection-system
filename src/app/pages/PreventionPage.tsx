import { motion } from 'motion/react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Shield, Syringe, Wind, Droplets, Users, Home } from 'lucide-react';
import Navigation from '../components/Navigation';
import { useLanguage } from '../context/LanguageContext';
import { getTranslation } from '../translations/translations';
import { getComprehensiveTranslation } from '../translations/comprehensive-translations';

export default function PreventionPage() {
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
          <h1 className="text-4xl mb-4 text-gray-900">{t.prevention.title}</h1>
          <p className="text-xl text-gray-600 mb-12">
            {t.prevention.subtitle}
          </p>
        </motion.div>

        <div className="grid sm:grid-cols-2 gap-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <Card className="h-full">
              <CardHeader>
                <div className="w-14 h-14 bg-teal-100 rounded-full flex items-center justify-center mb-3">
                  <Shield className="h-7 w-7 text-teal-600" />
                </div>
                <CardTitle>{comprehensiveT.prevention.maskTitle}</CardTitle>
              </CardHeader>
              <CardContent className="text-gray-700">
                <p className="mb-3">
                  {comprehensiveT.prevention.maskDesc}
                </p>
                <p className="text-sm text-teal-600">
                  ✓ {comprehensiveT.prevention.maskTip}
                </p>
              </CardContent>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Card className="h-full">
              <CardHeader>
                <div className="w-14 h-14 bg-blue-100 rounded-full flex items-center justify-center mb-3">
                  <Wind className="h-7 w-7 text-blue-600" />
                </div>
                <CardTitle>{comprehensiveT.prevention.ventilationTitle}</CardTitle>
              </CardHeader>
              <CardContent className="text-gray-700">
                <p className="mb-3">
                  {comprehensiveT.prevention.ventilationDesc}
                </p>
                <p className="text-sm text-blue-600">
                  ✓ {comprehensiveT.prevention.ventilationTip}
                </p>
              </CardContent>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <Card className="h-full">
              <CardHeader>
                <div className="w-14 h-14 bg-purple-100 rounded-full flex items-center justify-center mb-3">
                  <Droplets className="h-7 w-7 text-purple-600" />
                </div>
                <CardTitle>{comprehensiveT.prevention.coughTitle}</CardTitle>
              </CardHeader>
              <CardContent className="text-gray-700">
                <p className="mb-3">
                  {comprehensiveT.prevention.coughDesc}
                </p>
                <p className="text-sm text-purple-600">
                  ✓ {comprehensiveT.prevention.coughTip}
                </p>
              </CardContent>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <Card className="h-full">
              <CardHeader>
                <div className="w-14 h-14 bg-green-100 rounded-full flex items-center justify-center mb-3">
                  <Syringe className="h-7 w-7 text-green-600" />
                </div>
                <CardTitle>{comprehensiveT.prevention.vaccineTitle}</CardTitle>
              </CardHeader>
              <CardContent className="text-gray-700">
                <p className="mb-3">
                  {comprehensiveT.prevention.vaccineDesc}
                </p>
                <p className="text-sm text-green-600">
                  ✓ {comprehensiveT.prevention.vaccineTip}
                </p>
              </CardContent>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.5 }}
          >
            <Card className="h-full">
              <CardHeader>
                <div className="w-14 h-14 bg-amber-100 rounded-full flex items-center justify-center mb-3">
                  <Users className="h-7 w-7 text-amber-600" />
                </div>
                <CardTitle>{comprehensiveT.prevention.testTitle}</CardTitle>
              </CardHeader>
              <CardContent className="text-gray-700">
                <p className="mb-3">
                  {comprehensiveT.prevention.testDesc}
                </p>
                <p className="text-sm text-amber-600">
                  ✓ {comprehensiveT.prevention.testTip}
                </p>
              </CardContent>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.6 }}
          >
            <Card className="h-full">
              <CardHeader>
                <div className="w-14 h-14 bg-red-100 rounded-full flex items-center justify-center mb-3">
                  <Home className="h-7 w-7 text-red-600" />
                </div>
                <CardTitle>{comprehensiveT.prevention.healthyTitle}</CardTitle>
              </CardHeader>
              <CardContent className="text-gray-700">
                <p className="mb-3">
                  {comprehensiveT.prevention.healthyDesc}
                </p>
                <p className="text-sm text-red-600">
                  ✓ {comprehensiveT.prevention.healthyTip}
                </p>
              </CardContent>
            </Card>
          </motion.div>
        </div>

        {/* Additional Tips */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.7 }}
          className="mt-8"
        >
          <Card className="bg-teal-50 border-teal-200">
            <CardHeader>
              <CardTitle className="text-teal-900">{comprehensiveT.prevention.rememberTitle}</CardTitle>
            </CardHeader>
            <CardContent className="text-teal-900 space-y-2">
              <p>• {comprehensiveT.prevention.remember1}</p>
              <p>• {comprehensiveT.prevention.remember2}</p>
              <p>• {comprehensiveT.prevention.remember3}</p>
              <p>• {comprehensiveT.prevention.remember4}</p>
            </CardContent>
          </Card>
        </motion.div>
      </main>
    </div>
  );
}