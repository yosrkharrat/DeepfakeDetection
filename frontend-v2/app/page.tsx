import { Hero } from "@/components/landing/hero"
import { FeatureCards } from "@/components/landing/feature-cards"
import { ModelSection } from "@/components/landing/model-section"
import { HowItWorks } from "@/components/landing/how-it-works"
import { CTASection } from "@/components/landing/cta-section"
import { Footer } from "@/components/landing/footer"

export default function LandingPage() {
  return (
    <main>
      <Hero />
      <FeatureCards />
      <ModelSection />
      <HowItWorks />
      <CTASection />
      <Footer />
    </main>
  )
}
