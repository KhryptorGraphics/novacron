describe('Dashboard UI', () => {
  it('loads and displays metrics', () => {
    cy.visit('/dashboard');
    cy.contains('System Status').should('be.visible');
    cy.get('[data-test-id="cpu-usage"]').should('exist');
  });
});
