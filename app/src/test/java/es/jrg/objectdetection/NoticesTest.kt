package es.jrg.objectdetection

import org.junit.Assert.assertTrue
import org.junit.Test

class NoticesTest {
    @Test
    fun everyNoticeHasANameALicenceAndAnHttpsLink() {
        assertTrue(Notices.all.isNotEmpty())
        for (n in Notices.all) {
            assertTrue(n.name.isNotBlank())
            assertTrue(n.licence.isNotBlank())
            assertTrue(n.url.startsWith("https://"))
        }
    }

    @Test
    fun theLicenceStatementSaysItIsNotOpenSource() {
        assertTrue("not an open-source licence" in LICENSE_STATEMENT)
        assertTrue("PolyForm Noncommercial" in LICENSE_STATEMENT)
    }

    @Test
    fun theLinksAreHttps() {
        assertTrue(PRIVACY_POLICY_URL.startsWith("https://"))
        assertTrue(SOURCE_URL.startsWith("https://"))
    }
}
